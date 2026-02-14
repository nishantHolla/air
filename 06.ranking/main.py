import sys
import random
from pathlib import Path
import time

from inverted_index import InvertedIndex, T_document
from usage import print_usage
from utils import (
    index_file,
    database_file,
    read_dataset,
    get_query_from_user,
    term_frequency_file,
    document_frequency_file,
    inverse_document_frequency_file,
)

selected_topics: list[str] = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
]
docs_per_topic: int = 100


def index(dataset_file: Path | str, output_dir: Path | str) -> int:
    if not Path(dataset_file).is_file():
        return 1

    if not Path(output_dir).is_dir():
        return 2

    documents = read_dataset(dataset_file)

    ir.index(documents)
    ir.save(
        {
            "index_file": index_file(output_dir),
            "database_file": database_file(output_dir),
        }
    )

    return 0


def query(input_dir: Path | str) -> int:
    if not Path(input_dir).is_dir():
        return 1

    ir.load(
        {
            "index_file": index_file(input_dir),
            "database_file": database_file(input_dir),
        }
    )
    op, a, b = get_query_from_user()

    result = ir.query(op, a, b)
    for r in result:
        print(f"{r['name']}\n{r['text']}\n\n")

    print(f"Found {len(result)} results")
    return 0


def phrased_query(input_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    ir.load(
        {"index_file": index_file(input_dir), "database_file": database_file(input_dir)}
    )
    q = input("Enter phrased query:")

    result = ir.phrased_query(q)
    for r in result:
        print(f"{r['name']}\n{r['text']}\n\n")

    print(f"Found {len(result)} results")
    return 0


def tolerant_query(input_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    ir.load(
        {"index_file": index_file(input_dir), "database_file": database_file(input_dir)}
    )
    q = input("Enter tolerant query:")
    t = int(input("Enter tolerance value (0: No tolerance): "))

    result, tolerance = ir.tolerant_query(q, t)
    for r in result:
        print(f"{r['name']}\n{r['text']}\n\n")

    print(f"Found {len(result)} results")

    if len(tolerance) > 0:
        print("Had tolerance for terms: ")

    for t in tolerance:
        print("\t" + t)

    print()
    return 0


def tfidf(dataset_file: Path | str, output_dir: Path | str) -> int:
    if not Path(dataset_file).is_file():
        return 1

    if not Path(output_dir).is_dir():
        return 2

    documents: list[T_document] = read_dataset(dataset_file)

    print("Selected topics: ")
    for topic in selected_topics:
        print(f"\t{topic}")
    print("Documents per topic: ", docs_per_topic)

    ir.index(documents)
    ir.tfidf(documents)
    ir.save(
        {
            "index_file": index_file(output_dir),
            "database_file": database_file(output_dir),
            "term_frequency_file": term_frequency_file(output_dir),
            "document_frequency_file": document_frequency_file(output_dir),
            "inverse_document_frequency_file": inverse_document_frequency_file(
                output_dir
            ),
        }
    )

    return 0


def test(test_file: Path | str, dataset_file: Path | str) -> int:
    if not Path(test_file).is_file():
        return 1

    with open(test_file, "r", encoding="latin-1") as f:
        tests = f.readlines()

    if not Path(dataset_file).is_file():
        return 2

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

    for test in tests:
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
        f"Index time: {index_time:.8f}s for {len(documents)} documents"
        f" => {index_time / len(documents):.8f}s per doc"
    )

    print(
        f"Query time: {query_time:.8f}s for {len(tests)} tests"
        f" => {query_time / len(tests):.8f}s per test"
    )
    for o, v in operations.items():
        print(
            f"     {o} operation: {v['time']:.8f}s for {v['count']} operations"
            f" => {v['time'] / v['count']:.8f}s per operation"
        )
    print()

    return 0


def generate_test(input_dir: Path | str, count: int, test_file: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    if not Path(test_file).parent.is_dir():
        return 2

    ir.load(
        {"index_file": index_file(input_dir), "database_file": database_file(input_dir)}
    )
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

    return 0


def test_tokenizer(dataset_file: Path | str) -> int:
    if not Path(dataset_file).is_file():
        return 1

    documents = read_dataset(dataset_file)
    print(documents[0]["text"])
    print()
    print(ir._tokenize(documents[0]["text"]))

    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        exit(1)

    ir = InvertedIndex()
    ir.info()

    if sys.argv[1] == "index":
        if len(sys.argv) != 4:
            print_usage("index")
            exit(1)

        dataset_file = sys.argv[2]
        output_dir = sys.argv[3]
        ec: int = index(dataset_file, output_dir)

        if ec == 1:
            print(f"Failed to find file {dataset_file}")
            exit(ec)

        elif ec == 2:
            print(f"Failed to find directory {output_dir}")
            exit(ec)

    elif sys.argv[1] == "query":
        if len(sys.argv) != 3:
            print_usage("query")
            exit(1)

        index_dir = Path(sys.argv[2])
        ec = query(index_dir)

        if ec == 1:
            print(f"Failed to find directory {index_dir}")
            exit(ec)

    elif sys.argv[1] == "phrased_query":
        if len(sys.argv) != 3:
            print_usage("query")
            exit(1)

        index_dir = Path(sys.argv[2])
        ec = phrased_query(index_dir)

        if ec == 1:
            print(f"Failed to find directory {index_dir}")
            exit(ec)

    elif sys.argv[1] == "tolerant_query":
        if len(sys.argv) != 3:
            print_usage("tolerant_query")
            exit(1)

        index_dir = Path(sys.argv[2])
        ec = tolerant_query(index_dir)

        if ec == 1:
            print(f"Failed to find directory {index_dir}")
            exit(ec)

    elif sys.argv[1] == "tfidf":
        if len(sys.argv) != 4:
            print_usage("tfidf")
            exit(1)

        dataset_file = sys.argv[2]
        output_dir = sys.argv[3]
        ec: int = tfidf(dataset_file, output_dir)

        if ec == 1:
            print(f"Failed to find file {dataset_file}")
            exit(ec)

        elif ec == 2:
            print(f"Failed to find directory {output_dir}")
            exit(ec)

    elif sys.argv[1] == "tfidf_query":
        if len(sys.argv) != 3:
            print_usage("tfidf_query")
            exit(1)

        index_dir = Path(sys.argv[2])
        ir.load(
            {
                "index_file": index_file(index_dir),
                "database_file": database_file(index_dir),
                "term_frequency_file": term_frequency_file(index_dir),
                "document_frequency_file": document_frequency_file(index_dir),
                "inverse_document_frequency_file": inverse_document_frequency_file(
                    index_dir
                ),
            }
        )

        q = input("Enter query: ")
        result = ir.tfidf_query(q)
        for d in result:
            r, s = d
            print(f"==============================> Name: {r['name']} Score: {s}\n")
            print(f"{r['text'].strip()}\n")

    elif sys.argv[1] == "time":
        if len(sys.argv) != 4:
            print_usage("time")
            exit(1)

        test_file = Path(sys.argv[2])
        dataset_file = Path(sys.argv[3])
        ec = test(test_file, dataset_file)

        if ec == 1:
            print(f"Failed to find file {test_file}")
            exit(ec)

        elif ec == 2:
            print(f"Failed to find file {dataset_file}")
            exit(ec)

    elif sys.argv[1] == "generate_test":
        if len(sys.argv) != 5:
            print_usage("generate_test")
            exit(1)

        index_dir = Path(sys.argv[2])

        try:
            count = int(sys.argv[3])
        except ValueError:
            print(f"Invalid count {sys.argv[3]}")
            exit(2)

        test_file = Path(sys.argv[4])

        ec = generate_test(index_dir, count, test_file)

        if ec == 1:
            print(f"Failed to find directory {index_dir}")
            exit(ec)

        elif ec == 2:
            print(f"Failed to find directory {Path(test_file).parent}")
            exit(ec)

    elif sys.argv[1] == "test_tokenizer":
        if len(sys.argv) != 3:
            print_usage("test_tokenizer")
            exit(1)

        dataset_file = Path(sys.argv[2])

        ec = test_tokenizer(dataset_file)

        if ec == 1:
            print(f"Failed to find file {dataset_file}")
            exit(ec)

    elif sys.argv[1] == "test_edit_distance":
        a = input("Enter first term: ")
        b = input("Enter first term: ")

        print("The edit distance is", ir._edit_distance(a, b))

    else:
        print_usage()
        exit(3)

    print(f"Vocabulary size: {len(ir.get_vocab())}")
