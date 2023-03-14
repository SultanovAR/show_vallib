import pandas as pd
import ast
from importlib import import_module


def get_callable_from_path(path: str) -> callable:
    ".".join(path.split(".")[:-1])
    func = getattr(import_module(
        ".".join(path.split(".")[:-1])), path.split(".")[-1])

    return func


def parse_pipeline(path_to_xlsx: str):
    """
    The parse_pipeline function reads the pipeline configuration from an Excel file.
    The function returns two dictionaries: tests_desc and agg_config.
    tests_desc is a dictionary of test descriptions, where each key is a test name (test_key) and each value is another dictionary with the following keys:
    
    :param path_to_xlsx: str: Specify the path to the excel file with tests and aggregation configuration
    :return: A dictionary of tests and a dictionary of aggregation blocks
    """
    full_pipeline_df = pd.read_excel(path_to_xlsx, sheet_name="tests_config")
    agg_config = pd.read_excel(
        path_to_xlsx, sheet_name="agg_config").set_index('block_key').to_dict('index')

    code_columns = ["test_key", "import_path",
                    "block_key", "params", "informative"]
    excel_columns = [
        "test_key",
        "Название",
        "Цель",
        "Интерпретация",
        "Границы красный",
        "Границы желтый",
        "Границы зеленый",
    ]

    tests_desc = (
        full_pipeline_df[set(code_columns) | set(
            excel_columns)].set_index("test_key").T.to_dict()
    )

    for test in tests_desc:
        tests_desc[test]["params"] = ast.literal_eval(
            tests_desc[test]["params"]
        )

    # pipeline_for_excel = (
    #     full_pipeline_df[excel_columns].set_index("test_key").T.to_dict()
    # )

    return tests_desc, agg_config
