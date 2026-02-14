from typing import TypedDict
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


class InvertedIndex:
    """Implementation of inverted index with linked list and hash-map"""

    def __init__(self):
        """Initialize the class object and setup nltk"""
        self._nltk_setup()
        self._index: T_index = dict()
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

    def _and(self, a: str, b: str) -> list[int]:
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

    def _or(self, a: str, b: str) -> list[int]:
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

    def _not(self, a: str) -> list[int]:
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

    def info(self) -> None:
        """Print information about the implementation of the class to stdout"""
        print(f"Remove stop words: {self.REMOVE_STOP_WORDS}")
        print(f"Perform Lemmatization: {self.LEMMATIZE}")
        print("Index data structure: HashMap")
        print("Posting data structure: Array")
        print(f"Use skip list: {self.USE_SKIP_LIST}")
        print(f"Use set operations: {self.USE_SET_OPERATIONS}")
        print("Implements phrased queries: Yes")
        print("Implements tolerant retrieval: No")

        print()

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
                if token in self._index:
                    self._index[token].append(
                        {"doc_id": document_id, "positions": t_map[token]}
                    )
                else:
                    self._index[token] = [
                        {"doc_id": document_id, "positions": t_map[token]}
                    ]

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

    def query(self, op: str, a: str, b: str | None = None) -> list[T_document]:
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
            result = self._and(a, b)

        elif op == "or" and b:
            result = self._or(a, b)

        elif op == "not":
            result = self._not(a)

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
            result (list[T_document]): Resulting list of documents that sattisfoy the condition
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

    def get_vocab(self) -> T_vocab:
        return self._vocab
