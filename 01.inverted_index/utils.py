from pathlib import Path
from inverted_index import T_document


def index_file(d: Path | str) -> Path:
    """
    Returns the expected path for the inverted index json file for the given directory

    Args:
        d (Path | str): Directory to append the file name to

    Returns:
        (Path): The path to the inverted index json file
    """
    return Path(d) / "inverted_index.json"


def database_file(d: Path | str) -> Path:
    """
    Returns the expected path for the database json file for the given directory

    Args:
        d (Path | str): Directory to append the file name to

    Returns:
        (Path): The path to the database json file
    """
    return Path(d) / "database.json"


def read_dataset(dataset_file: Path | str) -> list[T_document]:
    """
    Read the dataset file and parse it into a list of documents

    Args:
        database_file (Path | str): The path to the dataset file

    Returns:
        documents (list[T_document]): List of documents parsed
    """

    with open(dataset_file, "r", encoding="latin-1") as f:
        data = f.readlines()

    documents = []
    for line in data:
        movie_name, review = line.split("\t")
        documents.append({"name": movie_name.strip(), "review": review.strip()})

    return documents


def get_query_from_user() -> tuple[str, str, str | None]:
    """
    Prompt the user to input a query from stdin

    term_b is set to None if the opeartion is 'not'

    Returns
        (tuple[str, str, str | None]): Tuple containing (operation, term_a, term_b)
    """
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
