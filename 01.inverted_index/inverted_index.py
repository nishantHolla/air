import string
import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from node import Node

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

T_index = dict[str, dict[str, Node]]
T_document = dict[str, str]
T_database = list[T_document]
T_vocab = set[str]


class InvertedIndex:
    def __init__(self):
        self._index: T_index = dict()
        self._database: T_database = list()
        self._vocab: T_vocab = set()
        self._translator = str.maketrans("", "", string.punctuation)
        self.REMOVE_STOP_WORDS = True
        self.LEMMATIZE = True

    def _tokenize(self, text: str) -> set[str]:
        text = text.strip().translate(self._translator)
        tokens = word_tokenize(text)

        if self.REMOVE_STOP_WORDS:
            tokens = [w for w in tokens if w.lower() not in stop_words]

        if self.LEMMATIZE:
            tokens = [lemmatizer.lemmatize(w) for w in tokens]

        return set(tokens)

    def _and(self, a: str, b: str) -> list[int]:
        if a not in self._index or b not in self._index:
            return []

        result = []

        a_node = self._index[a]["nodes"]
        b_node = self._index[b]["nodes"]

        while a_node and b_node:
            if a_node.value == b_node.value:
                result.append(a_node.value)
                a_node = a_node.next
                b_node = b_node.next

            elif a_node.value < b_node.value:
                a_node = a_node.next
            else:
                b_node = b_node.next

        return result

    def _or(self, a: str, b: str) -> list[int]:
        if a not in self._index and b not in self._index:
            return []

        if a not in self._index:
            return Node.to_list(self._index[b]["nodes"])

        if b not in self._index:
            return Node.to_list(self._index[a]["nodes"])

        result = []

        a_node = self._index[a]["nodes"]
        b_node = self._index[b]["nodes"]

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
        if a not in self._vocab:
            return []

        a_node = self._index[a]["nodes"]
        result = []

        prev = 0
        while a_node and prev < len(self._database):
            result += list(range(prev, a_node.value))
            prev = a_node.value + 1
            a_node = a_node.next

        return result

    def index(self, documents: list[T_document]) -> None:
        db_size = len(self._database)
        for i, document in enumerate(documents):
            tokens = self._tokenize(document["review"])
            for t in tokens:
                self._vocab.add(t)
            document_id = db_size + i

            self._database.append(document)
            for token in tokens:
                n = Node(document_id)
                if token not in self._index:
                    self._index[token] = {"tail": n, "nodes": n}
                else:
                    self._index[token]["tail"].next = n
                    self._index[token]["tail"] = n

    def save(self, index_path: Path | str, database_path: Path | str) -> None:
        index = dict()
        for k, v in self._index.items():
            index[k] = Node.to_list(v["nodes"])

        with open(index_path, "w") as f:
            json.dump(index, f, indent=4)

        with open(database_path, "w") as f:
            json.dump(self._database, f, indent=4)

    def load(self, index_path: Path | str, database_path: Path | str) -> None:
        with open(index_path, "r") as f:
            index = json.load(f)

        for k, v in index.items():
            n = Node.from_list(v)
            if n:
                self._index[k] = {"tail": Node.get_tail(n), "nodes": n}

        self._vocab = set(self._index.keys())

        with open(database_path, "r") as f:
            self._database = json.load(f)

    def query(self, op: str, a: str, b: str | None = None) -> list[T_document]:
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
