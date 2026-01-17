import sys

INDEX_ACTION_USAGE = """
    index: Index the movies dataset to produce index.json and databaes.json
        arg0: Path to the dataset file
        arg1: Path to the directory that will contain the index.json and database.json
"""

QUERY_ACTION_USAGE = """
    query: Prompt for a query and then execute the query by fetching relavent information from the database
        arg0: Path to the directory that contains the index.json and database.json
"""

TIME_ACTION_USAGE = """
    time: Time the indexing and query function based on a set of test queries read from a file
        arg0: Path to the test.txt file
        arg1: Path to the dataset file
"""

GENERATE_TEST_USAGE = """
    generate_test: Generate a test.txt file with given number of test cases
        arg0: Path to the directory that contains the index.json and database.json
        arg1: Count of number of test cases to generate
        arg1: Path to the test.txt file to be generated
"""

TEST_TOKENIZER_ACTION_USAGE = """
    test_tokenizer: View the output of the tokenizer by tokenizing the first data point of the database
        arg0: Path to the dataset file
"""

action_usage_map = {
    "index": INDEX_ACTION_USAGE,
    "query": QUERY_ACTION_USAGE,
    "time": TIME_ACTION_USAGE,
    "generate_test": GENERATE_TEST_USAGE,
    "test_tokenizer": TEST_TOKENIZER_ACTION_USAGE,
}


def print_usage(action: str | None = None):
    if not action or action not in action_usage_map:
        print(f"""Usage: {sys.argv[0]} {action} [args]
actions:
{INDEX_ACTION_USAGE}
{QUERY_ACTION_USAGE}
{TIME_ACTION_USAGE}
{GENERATE_TEST_USAGE}
{TEST_TOKENIZER_ACTION_USAGE}
""")
    else:
        print(f"""Usage: {sys.argv[0]} {action} [args]
{action_usage_map[action]}
""")
