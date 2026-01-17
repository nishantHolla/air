import sys
import random
from inverted_index import InvertedIndex
from pathlib import Path
from usage import print_usage
import time


def index_file(d: Path | str) -> Path:
    return Path(d) / "inverted_index.json"


def database_file(d: Path | str) -> Path:
    return Path(d) / "database.json"


def read_dataset(dataset_file: Path | str) -> list[dict[str, str]]:
    with open(dataset_file, "r", encoding="latin-1") as f:
        data = f.readlines()

    documents = []
    for line in data:
        movie_name, review = line.split("\t")
        documents.append({"name": movie_name.strip(), "review": review.strip()})

    return documents


def get_query_from_user() -> tuple[str, str, str | None]:
    while True:
        op = input("Enter operation: ")
        if op not in ["and", "not", "or"]:
            print("Invalid operation")
            continue

        break

    a = input("Enter first term: ")
    if op != "not":
        b = input("Enter second term: ")
    else:
        b = None

    return (op, a, b)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        exit(1)

    ir = InvertedIndex()

    if sys.argv[1] == "index":
        if len(sys.argv) != 4:
            print_usage("index")
            exit(1)

        dataset_file = Path(sys.argv[2])
        if not dataset_file.is_file():
            print(f"Failed to find file {dataset_file}")
            exit(2)

        output_dir = Path(sys.argv[3])
        if not output_dir.is_dir():
            print(f"Failed to find directory {output_dir}")

        documents = read_dataset(dataset_file)
        i_file = index_file(output_dir)
        d_file = database_file(output_dir)

        ir.index(documents)
        ir.save(i_file, d_file)

    elif sys.argv[1] == "query":
        if len(sys.argv) != 3:
            print_usage("query")
            exit(1)

        i_dir = Path(sys.argv[2])
        if not i_dir.is_dir():
            print(f"Failed to find directory {i_dir}")

        i_file = index_file(i_dir)
        d_file = database_file(i_dir)

        ir.load(i_file, d_file)
        op, a, b = get_query_from_user()

        result = ir.query(op, a, b)
        for r in result:
            print(f"{r['name']}\n{r['review']}\n\n")

        print(f"Found {len(result)} results")

    elif sys.argv[1] == "time":
        if len(sys.argv) != 4:
            print_usage("time")
            exit(1)

        test_file = Path(sys.argv[2])
        if not test_file.is_file():
            print(f"Failed to find file {test_file}")
            exit(2)

        with open(test_file, "r", encoding="latin-1") as f:
            tests = f.readlines()

        dataset_file = Path(sys.argv[3])
        if not dataset_file.is_file():
            print(f"Failed to find file {dataset_file}")
            exit(2)

        documents = read_dataset(dataset_file)

        index_start = time.perf_counter()
        ir.index(documents)
        index_end = time.perf_counter()
        index_time = index_end - index_start

        query_time = 0
        operations = {
            "and": {"time": 0.0, "count": 0},
            "or": {"time": 0.0, "count": 0},
            "not": {"time": 0.0, "count": 0},
        }

        for i, test in enumerate(tests):
            test = test.strip().split(" ")

            if len(test) == 2:
                a, o = test
                b = None
            else:
                a, o, b = test

            query_start = time.perf_counter()
            ir.query(o, a, b)
            query_end = time.perf_counter()
            query_time += query_end - query_start

            operations[o]["time"] += query_end - query_start
            operations[o]["count"] += 1

        print(
            f"Index time: {index_time:.4f}s for {len(documents)} documents"
            f" => {index_time / len(documents)}s per doc"
        )

        print(
            f"Query time: {query_time:.4f}s for {len(tests)} tests"
            f" => {query_time / len(tests)}s per test"
        )
        for o, v in operations.items():
            print(
                f"     {o} operation: {v['time']}s for {v['count']} operations"
                f" => {v['time'] / v['count']}s per operation"
            )

    elif sys.argv[1] == "generate_test":
        if len(sys.argv) != 5:
            print_usage("time")
            exit(1)

        i_dir = Path(sys.argv[2])
        if not i_dir.is_dir():
            print(f"Failed to find directory {i_dir}")
            exit(2)

        try:
            count = int(sys.argv[3])
        except ValueError:
            print(f"Invalid count {sys.argv[3]}")
            exit(2)

        test_file = Path(sys.argv[4])
        if not test_file.parent.is_dir():
            print(f"Failed to find directory {test_file.parent}")
            exit(2)

        i_file = index_file(i_dir)
        d_file = database_file(i_dir)

        ir.load(i_file, d_file)
        vocab = list(ir.get_vocab())
        operations = ["and", "or", "not"]

        tests = []

        for _ in range(count):
            o = random.choice(operations)
            a = random.choice(vocab)
            if o != "not":
                b = random.choice(vocab)
            else:
                b = ""
            tests.append(f"{a} {o} {b}")

        with open(test_file, "w", encoding="latin-1") as f:
            f.write("\n".join(tests))

    elif sys.argv[1] == "test_tokenizer":
        if len(sys.argv) != 3:
            print_usage("test_tokenizer")
            exit(1)

        dataset_file = Path(sys.argv[2])
        if not dataset_file.is_file():
            print(f"Failed to find file {dataset_file}")
            exit(2)

        documents = read_dataset(dataset_file)

        print(documents[0]["review"])
        print()
        print(ir._tokenize(documents[0]["review"]))

    else:
        print_usage()
        exit(3)
