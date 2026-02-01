import math
from typing import TypedDict
import string
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from list_node import ListNode


class T_posting(TypedDict):
    tail: ListNode
    nodes: ListNode
    length: int


class T_document(TypedDict):
    name: str
    review: str


T_index = dict[str, T_posting]
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
        self.USE_SKIP_LIST: bool = True

    def _nltk_setup(self) -> None:
        """Download required nltk modules and setup stopwords and lemmatizer"""
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)

        self._stop_words: set[str] = set(stopwords.words("english"))
        self._lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

    def _tokenize(self, input_text: str) -> set[str]:
        """
        Tokenize a given string of text

        Args:
            text (str): Text to tokenize

        Returns:
            tokens (set[str]): Set of tokens
        """
        text: str = input_text.strip().translate(self._translator)
        tokens: list[str] = word_tokenize(text)

        if self.REMOVE_STOP_WORDS:
            tokens = [w for w in tokens if w.lower() not in self._stop_words]

        if self.LEMMATIZE:
            tokens = [self._lemmatizer.lemmatize(w) for w in tokens]

        return set(tokens)

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

        a_node: ListNode | None = self._index[a]["nodes"]
        b_node: ListNode | None = self._index[b]["nodes"]

        while a_node and b_node:
            if a_node.value == b_node.value:
                result.append(a_node.value)
                a_node = a_node.next
                b_node = b_node.next

            elif a_node.value < b_node.value:
                if (
                    self.USE_SKIP_LIST
                    and a_node.skip
                    and a_node.skip.value < b_node.value
                ):
                    a_node = a_node.skip
                else:
                    a_node = a_node.next
            else:
                if (
                    self.USE_SKIP_LIST
                    and b_node.skip
                    and b_node.skip.value < a_node.value
                ):
                    b_node = b_node.skip
                else:
                    b_node = b_node.next

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
            return ListNode.to_list(self._index[b]["nodes"])

        if b not in self._index:
            return ListNode.to_list(self._index[a]["nodes"])

        result: list[int] = []

        a_node: ListNode | None = self._index[a]["nodes"]
        b_node: ListNode | None = self._index[b]["nodes"]

        while a_node and b_node:
            if a_node.value == b_node.value:
                result.append(a_node.value)
                a_node = a_node.next
                b_node = b_node.next

            elif a_node.value < b_node.value:
                result.append(a_node.value)
                a_node = a_node.next

            else:
                result.append(b_node.value)
                b_node = b_node.next

        while a_node:
            result.append(a_node.value)
            a_node = a_node.next

        while b_node:
            result.append(b_node.value)
            b_node = b_node.next

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

        a_node: ListNode | None = self._index[a]["nodes"]
        result: list[int] = []

        prev: int = 0
        N: int = len(self._database)

        while a_node:
            if prev < a_node.value:
                result.extend(range(prev, a_node.value))
            prev = a_node.value + 1
            a_node = a_node.next

        if prev < N:
            result.extend(range(prev, N))

        return result

    def info(self) -> None:
        """Print information about the implementation of the class to stdout"""
        print(f"Remove stop words: {self.REMOVE_STOP_WORDS}")
        print(f"Perform Lemmatization: {self.LEMMATIZE}")
        print("Index data structure: HashMap")
        print("Posting data structure: Linked List")
        print(f"Use skip list: {self.USE_SKIP_LIST}")
        print("Implements phrased queries: No")
        print("Implements tolerant retrieval: No")

        print()

    def index(self, documents: list[T_document]) -> None:
        """
        Index a list of documents to build its postings

        Args:
            documents (list[T_document]): List of documents in the form of T_document dict
        """
        db_size: int = len(self._database)
        for i, document in enumerate(documents):
            tokens: set[str] = self._tokenize(document["review"])
            for t in tokens:
                self._vocab.add(t)
            document_id: int = db_size + i

            self._database.append(document)
            for token in tokens:
                n: ListNode = ListNode(document_id)
                if token not in self._index:
                    self._index[token] = {"tail": n, "nodes": n, "length": 1}
                else:
                    self._index[token]["tail"].next = n
                    self._index[token]["tail"] = n
                    self._index[token]["length"] += 1

        if self.USE_SKIP_LIST:
            for _, v in self._index.items():
                skip_n: int = math.floor(math.sqrt(v["length"]))
                ListNode.make_skips(v["nodes"], skip_n)

    def save(self, index_path: Path | str, database_path: Path | str) -> None:
        """
        Save the current state of inverted index and database to a json file

        It is assumed that parent directory of both the paths are present. The file themselves
        need not be present

        Args:
            index_path (Path | str): Path to the json file to store the inverted index
            database_path (Path | str): Path to the json file to store the database
        """
        index: dict[str, list[int]] = dict()
        for k, v in self._index.items():
            index[k] = ListNode.to_list(v["nodes"])

        with open(index_path, "w") as f:
            json.dump(index, f, indent=4)

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
            index = json.load(f)

        for k, v in index.items():
            head, tail, length = ListNode.from_list(v)
            if head and tail:
                self._index[k] = {"tail": tail, "nodes": head, "length": length}
                if self.USE_SKIP_LIST:
                    skip_n: int = math.floor(math.sqrt(length))
                    ListNode.make_skips(head, skip_n)

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
            resutl (list[T_document]): Resulting list of documents that satisfy the condition
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

    def get_vocab(self) -> T_vocab:
        return self._vocab
