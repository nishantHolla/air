from typing import TypedDict, Optional
import string
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class T_document(TypedDict):
    name: str
    review: str


class T_posting(TypedDict):
    doc_id: int
    positions: list[int]


T_index = dict[str, list[T_posting]]
T_database = list[T_document]
T_vocab = set[str]


class BTreeNode:
    """Node in a B-tree"""

    def __init__(self, t: int, leaf: bool = False):
        """
        Initialize a B-tree node

        Args:
            t (int): Minimum degree (minimum number of keys is t-1)
            leaf (bool): True if node is a leaf node
        """
        self.keys: list[str] = []  # Keys (terms)
        self.values: list[list[T_posting]] = []  # Values (posting lists)
        self.children: list["BTreeNode"] = []  # Child pointers
        self.leaf: bool = leaf
        self.t: int = t  # Minimum degree
        self.n: int = 0  # Current number of keys

    def search(self, key: str) -> Optional[list[T_posting]]:
        """
        Search for a key in the subtree rooted at this node

        Args:
            key (str): The key to search for

        Returns:
            Optional[list[T_posting]]: The posting list if found, None otherwise
        """
        i = 0
        while i < self.n and key > self.keys[i]:
            i += 1

        if i < self.n and key == self.keys[i]:
            return self.values[i]

        if self.leaf:
            return None

        return self.children[i].search(key)

    def insert_non_full(self, key: str, value: list[T_posting]) -> None:
        """
        Insert a key-value pair into a node that is not full

        Args:
            key (str): The key to insert
            value (list[T_posting]): The value to insert
        """
        i = self.n - 1

        if self.leaf:
            # Insert key-value at the correct position
            self.keys.append("")
            self.values.append([])
            while i >= 0 and key < self.keys[i]:
                self.keys[i + 1] = self.keys[i]
                self.values[i + 1] = self.values[i]
                i -= 1
            self.keys[i + 1] = key
            self.values[i + 1] = value
            self.n += 1
        else:
            # Find child to insert into
            while i >= 0 and key < self.keys[i]:
                i -= 1
            i += 1

            # Check if child is full
            if self.children[i].n == 2 * self.t - 1:
                self.split_child(i)
                if key > self.keys[i]:
                    i += 1

            self.children[i].insert_non_full(key, value)

    def split_child(self, i: int) -> None:
        """
        Split the i-th child of this node

        Args:
            i (int): Index of the child to split
        """
        t = self.t
        y = self.children[i]
        z = BTreeNode(t, y.leaf)
        z.n = t - 1

        # Copy the last (t-1) keys and values of y to z
        z.keys = y.keys[t : 2 * t - 1]
        z.values = y.values[t : 2 * t - 1]

        # Store median key and value
        median_key = y.keys[t - 1]
        median_value = y.values[t - 1]

        # Keep first (t-1) keys and values in y
        y.keys = y.keys[: t - 1]
        y.values = y.values[: t - 1]

        # Copy the last t children of y to z if not leaf
        if not y.leaf:
            z.children = y.children[t : 2 * t]
            y.children = y.children[:t]

        # Move median key up to this node
        self.keys.insert(i, median_key)
        self.values.insert(i, median_value)
        self.children.insert(i + 1, z)
        self.n += 1

        # Update counts
        y.n = t - 1

    def traverse(self) -> list[tuple[str, list[T_posting]]]:
        """
        Traverse the subtree rooted at this node in sorted order

        Returns:
            list[tuple[str, list[T_posting]]]: List of (key, value) pairs
        """
        result = []
        i = 0
        for i in range(self.n):
            if not self.leaf:
                result.extend(self.children[i].traverse())
            result.append((self.keys[i], self.values[i]))

        if not self.leaf:
            result.extend(self.children[self.n].traverse())

        return result


class BTree:
    """B-tree implementation for inverted index"""

    def __init__(self, t: int = 3):
        """
        Initialize an empty B-tree

        Args:
            t (int): Minimum degree (minimum number of keys is t-1)
        """
        self.root: Optional[BTreeNode] = None
        self.t: int = t

    def search(self, key: str) -> Optional[list[T_posting]]:
        """
        Search for a key in the B-tree

        Args:
            key (str): The key to search for

        Returns:
            Optional[list[T_posting]]: The posting list if found, None otherwise
        """
        if self.root is None:
            return None
        return self.root.search(key)

    def insert(self, key: str, value: list[T_posting]) -> None:
        """
        Insert a key-value pair into the B-tree

        Args:
            key (str): The key to insert
            value (list[T_posting]): The value to insert
        """
        if self.root is None:
            self.root = BTreeNode(self.t, True)
            self.root.keys.append(key)
            self.root.values.append(value)
            self.root.n = 1
        else:
            # If root is full, split it
            if self.root.n == 2 * self.t - 1:
                s = BTreeNode(self.t, False)
                s.children.append(self.root)
                s.split_child(0)
                self.root = s

            self.root.insert_non_full(key, value)

    def traverse(self) -> list[tuple[str, list[T_posting]]]:
        """
        Traverse the B-tree in sorted order

        Returns:
            list[tuple[str, list[T_posting]]]: List of (key, value) pairs
        """
        if self.root is None:
            return []
        return self.root.traverse()

    def to_dict(self) -> dict[str, list[T_posting]]:
        """
        Convert B-tree to dictionary

        Returns:
            dict[str, list[T_posting]]: Dictionary representation
        """
        return dict(self.traverse())


class InvertedIndex:
    """Implementation of inverted index with linked list and hash-map"""

    def __init__(self):
        """Initialize the class object and setup nltk"""
        self._nltk_setup()
        self._index: T_index = dict()
        self._index_bt: BTree = BTree(t=3)  # B-tree with minimum degree 3
        self._database: T_database = list()
        self._vocab: T_vocab = set()
        self._translator: dict[int, int | None] = str.maketrans(
            "", "", string.punctuation
        )
        self.REMOVE_STOP_WORDS: bool = True
        self.LEMMATIZE: bool = True
        self.USE_SKIP_LIST: bool = False  # Always False
        self.USE_SET_OPERATIONS: bool = False

    def _nltk_setup(self) -> None:
        """Download required nltk modules and setup stopwords and lemmatizer"""
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)

        self._stop_words: set[str] = set(stopwords.words("english"))
        self._lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

    def _tokenize(self, input_text: str) -> list[str]:
        """
        Tokenize a given string of text

        Args:
            text (str): Text to tokenize

        Returns:
            tokens (list[str]): list of tokens
        """
        text: str = input_text.strip().translate(self._translator)
        tokens: list[str] = word_tokenize(text)

        if self.REMOVE_STOP_WORDS:
            tokens = [w for w in tokens if w.lower() not in self._stop_words]

        if self.LEMMATIZE:
            tokens = [self._lemmatizer.lemmatize(w) for w in tokens]

        return tokens

    def _and(self, a: str, b: str, use_btree=False) -> list[int]:
        """
        Performs 'and' operation on two given tokens to identify common documents that contain
        both the terms

        Args:
            a (str): First token
            b (str): Second token

        Returns:
            result (list[int]): List of document ids that is the intersection of the two postings
        """
        if a not in self._index or b not in self._index:
            return []

        result: list[int] = []

        if use_btree:
            a_search = self.search_btree(a)
            a_list: list[int] = [d["doc_id"] for d in a_search] if a_search else []

            b_search = self.search_btree(b)
            b_list: list[int] = [d["doc_id"] for d in b_search] if b_search else []

        else:
            a_list: list[int] = [k["doc_id"] for k in self._index[a]]
            b_list: list[int] = [k["doc_id"] for k in self._index[b]]

        if self.USE_SET_OPERATIONS:
            a_set: set[int] = set(a_list)
            b_set: set[int] = set(b_list)
            return sorted(list(a_set.intersection(b_set)))

        a_len: int = len(a_list)
        b_len: int = len(b_list)

        a_idx: int = 0
        b_idx: int = 0

        while a_idx < a_len and b_idx < b_len:
            if a_list[a_idx] == b_list[b_idx]:
                result.append(a_list[a_idx])
                a_idx += 1
                b_idx += 1

            elif a_list[a_idx] < b_list[b_idx]:
                a_idx += 1

            else:
                b_idx += 1

        return result

    def _or(self, a: str, b: str, use_btree=False) -> list[int]:
        """
        Performs 'or' operation on two given tokens to identify all documents that contain
        eiter of the terms

        Args:
            a (str): First token
            b (str): Second token

        Returns:
            result (list[int]): List of document ids that is the union of the two postings
        """
        if a not in self._index and b not in self._index:
            return []

        if a not in self._index:
            return [k["doc_id"] for k in self._index[b]]

        if b not in self._index:
            return [k["doc_id"] for k in self._index[a]]

        result: list[int] = []

        if use_btree:
            a_search = self.search_btree(a)
            a_list: list[int] = [d["doc_id"] for d in a_search] if a_search else []

            b_search = self.search_btree(b)
            b_list: list[int] = [d["doc_id"] for d in b_search] if b_search else []

        else:
            a_list: list[int] = [k["doc_id"] for k in self._index[a]]
            b_list: list[int] = [k["doc_id"] for k in self._index[b]]

        if self.USE_SET_OPERATIONS:
            a_set: set[int] = set(a_list)
            b_set: set[int] = set(b_list)
            return sorted(list(a_set.union(b_set)))

        a_len: int = len(a_list)
        b_len: int = len(b_list)

        a_idx: int = 0
        b_idx: int = 0

        while a_idx < a_len and b_idx < b_len:
            if a_list[a_idx] == b_list[b_idx]:
                result.append(a_list[a_idx])
                a_idx += 1
                b_idx += 1

            elif a_list[a_idx] < b_list[b_idx]:
                result.append(a_list[a_idx])
                a_idx += 1

            else:
                result.append(b_list[b_idx])
                b_idx += 1

        while a_idx < a_len:
            result.append(a_list[a_idx])
            a_idx += 1

        while b_idx < b_len:
            result.append(b_list[b_idx])
            b_idx += 1

        return result

    def _not(self, a: str, use_btree=False) -> list[int]:
        """
        Performs 'not' operation on the given token to identify documents that do not contain
        the terms

        Args:
            a (str): Token

        Returns:
            result (list[int]): List of document ids that is the complement of the postings
        """
        if a not in self._vocab:
            return list(range(len(self._database)))

        result: list[int] = []
        N: int = len(self._database)

        if use_btree:
            a_search = self.search_btree(a)
            a_list: list[int] = [d["doc_id"] for d in a_search] if a_search else []

        else:
            a_list: list[int] = [k["doc_id"] for k in self._index[a]]

        if self.USE_SET_OPERATIONS:
            univ: set[int] = set(range(0, N))
            a_set: set[int] = set(a_list)
            return sorted(list(univ.difference(a_set)))

        a_len: int = len(a_list)
        a_idx: int = 0

        prev: int = 0

        while a_idx < a_len:
            if prev < a_list[a_idx]:
                result.extend(range(prev, a_list[a_idx]))

            prev = a_list[a_idx] + 1
            a_idx += 1

        if prev < N:
            result.extend(range(prev, N))

        return result

    def _phrase_merge(
        self, a: list[T_posting], b: list[T_posting], base: int
    ) -> list[T_posting]:
        b_dict = {k["doc_id"]: k for k in b}
        result: list[T_posting] = []

        for posting in a:
            id: int = posting["doc_id"]
            target: set[int] = set([k + 1 + base for k in posting["positions"]])
            source: list[int] = b_dict[id]["positions"] if id in b_dict else []

            for i in source:
                if i in target:
                    result.append(posting)
                    break

        return result

    def _tolerant_merge(
        self, a: list[T_posting], b: list[T_posting]
    ) -> list[T_posting]:
        result: list[T_posting] = []

        a_len: int = len(a)
        b_len: int = len(b)

        a_idx: int = 0
        b_idx: int = 0

        while a_idx < a_len and b_idx < b_len:
            if a[a_idx]["doc_id"] == b[b_idx]["doc_id"]:
                result.append(a[a_idx])
                a_idx += 1

            elif a[a_idx]["doc_id"] < b[b_idx]["doc_id"]:
                result.append(a[a_idx])
                a_idx += 1

            else:
                result.append(b[b_idx])
                b_idx += 1

        while a_idx < a_len:
            result.append(a[a_idx])
            a_idx += 1

        while b_idx < b_len:
            result.append(b[b_idx])
            b_idx += 1

        return result

    def info(self) -> None:
        """Print information about the implementation of the class to stdout"""
        print(f"Remove stop words: {self.REMOVE_STOP_WORDS}")
        print(f"Perform Lemmatization: {self.LEMMATIZE}")
        print("Index data structure: HashMap + B-Tree")
        print("Posting data structure: Array")
        print(f"Use skip list: {self.USE_SKIP_LIST}")
        print(f"Use set operations: {self.USE_SET_OPERATIONS}")
        print("Implements phrased queries: Yes")
        print("Implements tolerant retrieval: Yes")

        print()

    def _edit_distance(self, a: str, b: str) -> int:
        mat = [[0 for _ in range(len(a) + 1)] for _ in range(len(b) + 1)]
        mat[0][0] = 0

        for j in range(len(a) + 1):
            mat[0][j] = j

        for i in range(len(b) + 1):
            mat[i][0] = i

        for i in range(1, len(b) + 1):
            for j in range(1, len(a) + 1):
                if b[i - 1] == a[j - 1]:
                    cost = 0
                else:
                    cost = 1

                mat[i][j] = min(
                    mat[i - 1][j - 1] + cost, mat[i - 1][j] + 1, mat[i][j - 1] + 1
                )

        return mat[len(b)][len(a)]

    def index(self, documents: list[T_document]) -> None:
        """
        Index a list of documents to build its postings

        Args:
            documents (list[T_document]): List of documents in the form of T_document dict
        """
        for document_id, document in enumerate(documents):
            tokens: list[str] = self._tokenize(document["review"])
            t_map: dict[str, list[int]] = dict()

            for i, t in enumerate(tokens):
                self._vocab.add(t)
                if t in t_map:
                    t_map[t].append(i)
                else:
                    t_map[t] = [i]

            self._database.append(document)
            for token in set(tokens):
                posting: T_posting = {"doc_id": document_id, "positions": t_map[token]}

                # Update hash map index
                if token in self._index:
                    self._index[token].append(posting)
                else:
                    self._index[token] = [posting]

                # Update B-tree index
                existing_postings = self._index_bt.search(token)
                if existing_postings is not None:
                    existing_postings.append(posting)
                else:
                    self._index_bt.insert(token, [posting])

    def save(self, index_path: Path | str, database_path: Path | str) -> None:
        """
        Save the current state of inverted index and database to a json file

        It is assumed that parent directory of both the paths are present. The file themselves
        need not be present

        Args:
            index_path (Path | str): Path to the json file to store the inverted index
            database_path (Path | str): Path to the json file to store the database
        """
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=4)

        with open(database_path, "w") as f:
            json.dump(self._database, f, indent=4)

    def load(self, index_path: Path | str, database_path: Path | str) -> None:
        """
        Load the state of inverted index and database from a json file
        Also rebuilds the B-tree index and verifies consistency

        It is assumed that both the json files are present

        Args:
            index_path (Path | str): Path to the json file to load the inverted index from
            database_path (Path | str): Path to the json file to load the database from
        """
        with open(index_path, "r") as f:
            self._index = json.load(f)

        self._vocab = set(self._index.keys())

        with open(database_path, "r") as f:
            self._database = json.load(f)

        # Rebuild B-tree from loaded hash map
        self._index_bt = BTree(t=3)
        for term, posting_list in self._index.items():
            self._index_bt.insert(term, posting_list)

        # Verify consistency
        if not self.verify_btree_consistency():
            raise ValueError(
                "B-tree and hash map indices are inconsistent after loading"
            )

    def query(
        self, op: str, a: str, b: str | None = None, use_btree=False
    ) -> list[T_document]:
        """
        Perform a query operation with the given terms to get the resultant posting list

        Empty list is returned on invalid operation or insufficient number of terms

        Args:
            op (str): Operation to perform
            a (str): First token
            b (str): Second token. Set to None if not required by the operation

        Returns:
            result (list[T_document]): Resulting list of documents that satisfy the condition
        """
        if op != "not" and b is None:
            return []

        result = []
        if op == "and" and b:
            result = self._and(a, b, use_btree)

        elif op == "or" and b:
            result = self._or(a, b, use_btree)

        elif op == "not":
            result = self._not(a, use_btree)

        documents = []
        for i in result:
            documents.append(self._database[i])

        return documents

    def phrased_query(self, query: str) -> list[T_document]:
        """
        Perform phrased query search with the given query string to get the resultant list of documents

        Empty list is returned on invalid operation

        Args:
            query (str): The query to perform

        Returns:
            result (list[T_document]): Resulting list of documents that sattisfy the condition
        """
        query_tokens: list[str] = self._tokenize(query)
        if len(query_tokens) == 0:
            return []

        result: list[T_posting] = self._index[query_tokens[0]]
        for i, token in enumerate(query_tokens[1:]):
            result = self._phrase_merge(result, self._index[token], i)

        documents: list[T_document] = []
        for i in result:
            documents.append(self._database[i["doc_id"]])

        return documents

    def tolerant_query(
        self, query: str, tolerance: int
    ) -> tuple[list[T_document], list[str]]:
        """Performs tolerant query search with the given query string to the resultant list of documents
        along with a list of strings it matched with the tolerance

        Empty list is returned on invalid operation

        Args:
            query (str)    : The query to perform
            tolerance (int): The tolerance of the search

        Returns:
            result (list[T_document], list[str]): Resulting list of documents that sattisfy the condition
                                                  and list of strings it matched as part of tolerance

        """
        terms: list[str] = [query]

        for word in self._vocab:
            if self._edit_distance(word, query) <= tolerance:
                terms.append(word)

        result: list[T_posting] = []
        for term in terms:
            result = self._tolerant_merge(result, self._index[term])

        documents: list[T_document] = []
        for i in result:
            documents.append(self._database[i["doc_id"]])

        return (documents, terms)

    def get_vocab(self) -> T_vocab:
        return self._vocab

    def search_btree(self, term: str) -> Optional[list[T_posting]]:
        """
        Search for a term in the B-tree index

        Args:
            term (str): The term to search for

        Returns:
            Optional[list[T_posting]]: The posting list if found, None otherwise
        """
        return self._index_bt.search(term)

    def get_btree_dict(self) -> dict[str, list[T_posting]]:
        """
        Get the B-tree index as a dictionary

        Returns:
            dict[str, list[T_posting]]: Dictionary representation of B-tree
        """
        return self._index_bt.to_dict()

    def verify_btree_consistency(self) -> bool:
        """
        Verify that the B-tree index and hash map index contain the same data

        Returns:
            bool: True if both indices are consistent, False otherwise
        """
        btree_dict = self._index_bt.to_dict()

        # Check if all keys match
        if set(btree_dict.keys()) != set(self._index.keys()):
            return False

        # Check if all values match
        for key in self._index:
            if btree_dict[key] != self._index[key]:
                return False

        return True
