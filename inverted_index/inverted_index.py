import string
import json
from pathlib import Path

T_index = dict[str, list[int]]
T_document = dict[str, str]
T_database = list[T_document]
T_vocab = list[str]


class InvertedIndex:
    def __init__(self):
        self._index: T_index = dict()
        self._database: T_database = list()
        self._vocab: T_vocab = list()

    def _tokenize(self, text: str) -> list[str]:
        text = text.strip()
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)

        return list(set(text.split(" ")))

    def _and(self, a: str, b: str) -> list[int]:
        a_set = set(self._index.get(a, []))
        b_set = set(self._index.get(b, []))

        return list(a_set.intersection(b_set))

    def _or(self, a: str, b: str) -> list[int]:
        a_set = set(self._index.get(a, []))
        b_set = set(self._index.get(b, []))

        return list(a_set.union(b_set))

    def _not(self, a: str) -> list[int]:
        d = range(0, len(self._database))
        return list(set(d) - set(self._index.get(a, [])))

    def index(self, documents: list[T_document]) -> None:
        db_size = len(self._database)
        for i, document in enumerate(documents):
            tokens = self._tokenize(document["review"])
            document_id = db_size + i

            self._database.append(document)
            for token in tokens:
                if token not in self._index:
                    self._index[token] = [document_id]
                else:
                    self._index[token].append(document_id)

    def save(self, index_path: Path, database_path: Path) -> None:
        with open(index_path, "w") as f:
            json.dump(self._index, f, indent=4)

        with open(database_path, "w") as f:
            json.dump(self._database, f, indent=4)

    def load(self, index_path: Path, database_path: Path) -> None:
        with open(index_path, "r") as f:
            self._index = json.load(f)

        with open(database_path, "r") as f:
            self._database = json.load(f)

    def query(self, op: str, a: str, b: str | None = None) -> T_document:
        if op != "not" and b is None:
            return []

        if op == "and":
            result = self._and(a, b)

        elif op == "or":
            result = self._or(a, b)

        elif op == "not":
            result = self._not(a)

        documents = []
        for i in result:
            documents.append(self._database[i])

        return documents
