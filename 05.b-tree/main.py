import sys
import random
from inverted_index import InvertedIndex
from pathlib import Path
from usage import print_usage
import time
from utils import index_file, database_file, read_dataset, get_query_from_user


def index(dataset_file: Path | str, output_dir: Path | str) -> int:
    if not Path(dataset_file).is_file():
        return 1

    if not Path(output_dir).is_dir():
        return 2

    documents = read_dataset(dataset_file)
    i_file = index_file(output_dir)
    d_file = database_file(output_dir)

    ir.index(documents)
    ir.save(i_file, d_file)

    return 0


def query(index_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    idx_file = index_file(index_dir)
    db_file = database_file(index_dir)

    ir.load(idx_file, db_file)
    op, a, b = get_query_from_user()

    result = ir.query(op, a, b)
    for r in result:
        print(f"{r['name']}\n{r['review']}\n\n")

    print(f"Found {len(result)} results")
    return 0


def btree(index_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    idx_file = index_file(index_dir)
    db_file = database_file(index_dir)

    ir.load(idx_file, db_file)
    op, a, b = get_query_from_user()

    result = ir.query(op, a, b, use_btree=True)
    if not result:
        return 0

    for r in result:
        print(f"{r['name']}\n{r['review']}\n\n")

    print(f"Found {len(result)} results")
    return 0


def phrased_query(index_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    idx_file = index_file(index_dir)
    db_file = database_file(index_dir)

    ir.load(idx_file, db_file)
    q = input("Enter phrased query:")

    result = ir.phrased_query(q)
    for r in result:
        print(f"{r['name']}\n{r['review']}\n\n")

    print(f"Found {len(result)} results")
    return 0


def tolerant_query(index_dir: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    idx_file = index_file(index_dir)
    db_file = database_file(index_dir)

    ir.load(idx_file, db_file)
    q = input("Enter tolerant query:")
    t = int(input("Enter tolerance value (0: No tolerance): "))

    result, tolerance = ir.tolerant_query(q, t)
    for r in result:
        print(f"{r['name']}\n{r['review']}\n\n")

    print(f"Found {len(result)} results")

    if len(tolerance) > 0:
        print("Had tolerance for terms: ")

    for t in tolerance:
        print("\t" + t)

    print()
    return 0


def test(test_file: Path | str, dataset_file: Path | str, use_btree=False) -> int:
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
        ir.query(o, a, b, use_btree=use_btree)
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


def generate_test(index_dir: Path | str, count: int, test_file: Path | str) -> int:
    if not Path(index_dir).is_dir():
        return 1

    if not Path(test_file).parent.is_dir():
        return 2

    i_file = index_file(index_dir)
    d_file = database_file(index_dir)

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

    return 0


def test_tokenizer(dataset_file: Path | str) -> int:
    if not Path(dataset_file).is_file():
        return 1

    documents = read_dataset(dataset_file)
    print(documents[0]["review"])
    print()
    print(ir._tokenize(documents[0]["review"]))

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

    elif sys.argv[1] == "btree":
        if len(sys.argv) != 3:
            print_usage("btree")
            exit(1)

        index_dir = Path(sys.argv[2])
        ec = btree(index_dir)

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

    elif sys.argv[1] == "time":
        if len(sys.argv) != 4:
            print_usage("time")
            exit(1)

        test_file = Path(sys.argv[2])
        dataset_file = Path(sys.argv[3])

        print("========== HASH MAP =============")

        ec = test(test_file, dataset_file, use_btree=False)

        if ec == 1:
            print(f"Failed to find file {test_file}")
            exit(ec)

        elif ec == 2:
            print(f"Failed to find file {dataset_file}")
            exit(ec)

        print("========== B TREE =============")

        ec = test(test_file, dataset_file, use_btree=True)

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
