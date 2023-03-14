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

    :param path_to_xlsx: str: Specify the path to the excel file with tests and aggregation configuration
    :return: 
        The function returns two dictionaries: tests_desc and agg_config.
        tests_desc is a dict with the following format:
            {
                'test_key': {
                    'import_path': test's import path. example:'sbe_vallib.table.general_tests.test_psi_factor'
                    'block_key': str,
                    'informative': int 0 or 1
                    'params': {
                        'param_1': param_value_1,
                        ...
                    }
                    'Название': tp.Optional[str],
                    'Цель': tp.Optional[str],
                    'Интерпретация': tp.Optional[str],
                    'Границы красный': tp.Optional[str],
                    'Границы желтый': tp.Optional[str],
                    'Границы зеленый': tp.Optional[str],
                    }
            }
        agg_config is a dict with the following format:
        {
            'block_key': {
                'print_name': 'Качество данных',
                'func': 'sbe_vallib.utils.report_helper.worst_semaphore'
            }
        }
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
