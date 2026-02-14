import math
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
    text: str


class T_posting(TypedDict):
    doc_id: int
    positions: list[int]


T_index = dict[str, list[T_posting]]
T_database = list[T_document]
T_vocab = set[str]
T_tf = dict[tuple[str, int], float]  # [term, doc_id] => score
T_df = dict[str, int]  # term => freq
T_idf = dict[str, float]  # term => score


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

        self._tf: T_tf = dict()
        self._df: T_df = dict()
        self._idf: T_idf = dict()

        self.REMOVE_STOP_WORDS: bool = True
        self.LEMMATIZE: bool = True
        self.USE_SKIP_LIST: bool = True
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

    def _build_tf(self, term: str, doc_id: int, tokens: list[str]) -> None:
        """
        Update the term-frequecy map for the given term, document id and the document text

        Args:
            term (str): Term to update
            doc_id (str): ID of the document to update
            text (str): Text of the document
        """
        self._tf[(term, doc_id)] = 1 + math.log10(tokens.count(term))

    def _build_df(self, term: str) -> None:
        """
        Update the document-frequency map for the given term

        Args:
            term (str): Term to update
        """
        if term in self._df:
            self._df[term] += 1
        else:
            self._df[term] = 1

    def _build_idf(self) -> None:
        """
        Update the inverse-document frequency map by using the document-frequency map
        """
        number_of_documents = len(self._database)

        for k, v in self._df.items():
            self._idf[k] = math.log10(number_of_documents / v)

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
        print("Index data structure: HashMap")
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
            tokens: list[str] = self._tokenize(document["text"])
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

    def tfidf(self, documents: list[T_document]) -> None:
        """
        Perform tfidf computation on list of documents

        Args:
            documents (list[T_documents]): List of documents in the form of T_document dict
        """
        for doc_id, document in enumerate(documents):
            tokens = self._tokenize(document["text"])
            visited: set[str] = set()
            self._database.append(document)

            for token in tokens:
                self._build_tf(token, doc_id, tokens)
                if token not in visited:
                    self._build_df(token)
                    visited.add(token)

        self._build_idf()

    def tfidf_query(
        self, query: str, k: int = 100000
    ) -> list[tuple[T_document, float]]:
        tokens: list[str] = self._tokenize(query)
        docs: dict[int, float] = dict()

        for token in tokens:
            for key, tf_score in self._tf.items():
                term, doc_id = key
                if term == token:
                    if doc_id in docs:
                        docs[doc_id] *= tf_score * self._idf[term]
                    else:
                        docs[doc_id] = tf_score * self._idf[term]

        ranked: list[tuple[float, int]] = [(value, key) for key, value in docs.items()]
        ranked.sort(reverse=True)
        length = min(len(ranked), k)

        result: list[tuple[T_document, float]] = []
        for d in ranked[:length]:
            result.append((self._database[d[1]], d[0]))

        return result

    def save(self, paths: dict[str, Path | str]) -> None:
        """
        Save the current state of inverted index and database to a json file

        It is assumed that parent directory of both the paths are present. The file themselves
        need not be present

        Args:
            paths (dict[str, Path | str]): Resources to save
        """
        if "index_file" in paths:
            with open(paths["index_file"], "w") as f:
                json.dump(self._index, f, indent=4)

        if "database_file" in paths:
            with open(paths["database_file"], "w") as f:
                json.dump(self._database, f, indent=4)

        if "term_frequency_file" in paths:
            d: dict[str, float] = dict()
            for k, v in self._tf.items():
                d[f"{k[0]}:{k[1]}"] = v

            with open(paths["term_frequency_file"], "w") as f:
                json.dump(d, f, indent=4)

        if "document_frequency_file" in paths:
            with open(paths["document_frequency_file"], "w") as f:
                json.dump(self._df, f, indent=4)

        if "inverse_document_frequency_file" in paths:
            with open(paths["inverse_document_frequency_file"], "w") as f:
                json.dump(self._idf, f, indent=4)

    def load(self, paths: dict[str, Path | str]) -> None:
        """
        Load the state of inverted index and database from a json file

        It is assumed that both the json files are present

        Args:
            paths (dict[str, Path | str]): Resources to load
        """
        if "index_file" in paths:
            with open(paths["index_file"], "r") as f:
                self._index = json.load(f)

            self._vocab = set(self._index.keys())

        if "database_file" in paths:
            with open(paths["database_file"], "r") as f:
                self._database = json.load(f)

        if "term_frequency_file" in paths:
            with open(paths["term_frequency_file"], "r") as f:
                d = json.load(f)

            for k, v in d.items():
                term, doc_id = k.split(":")
                self._tf[(term, int(doc_id))] = v

        if "document_frequency_file" in paths:
            with open(paths["document_frequency_file"], "r") as f:
                self._df = json.load(f)

        if "inverse_document_frequency_file" in paths:
            with open(paths["inverse_document_frequency_file"], "r") as f:
                self._idf = json.load(f)

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
