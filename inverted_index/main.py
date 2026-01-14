import sys
from inverted_index import InvertedIndex

DATASET_FILE = "./dataset/movies.txt"
INDEX_FILE = "./index/inverted_index.json"
DATABASE_FILE = "./index/database.json"


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} [index|query]")
    exit(1)

ir = InvertedIndex()

if sys.argv[1] == "index":
    with open(DATASET_FILE, "r", encoding="latin-1") as f:
        data = f.readlines()

    documents = []
    for i, line in enumerate(data):
        movie_name, review = line.split("\t")
        documents.append({"name": movie_name.strip(), "review": review.strip()})

    ir.index(documents)
    ir.save(INDEX_FILE, DATABASE_FILE)

elif sys.argv[1] == "query":
    ir.load(INDEX_FILE, DATABASE_FILE)

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

    result = ir.query(op, a, b)
    for r in result:
        print(f"{r['name']}\n{r['review']}\n\n")

elif sys.argv[1] == "test_tokenizer":
    with open(DATASET_FILE, "r", encoding="latin-1") as f:
        data = f.readlines()

    documents = []
    for i, line in enumerate(data):
        movie_name, review = line.split("\t")
        documents.append({"name": movie_name.strip(), "review": review.strip()})

    print(documents[0]["review"])
    print()
    print(ir._tokenize(documents[0]["review"]))
