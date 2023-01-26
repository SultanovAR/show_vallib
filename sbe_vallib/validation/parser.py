import pandas as pd
import ast
from importlib import import_module


def get_callable_from_path(path: str) -> callable:

    ".".join(path.split(".")[:-1])
    func = getattr(import_module(".".join(path.split(".")[:-1])), path.split(".")[-1])

    return func


def parse_pipeline(path_to_xlsx: str):

    full_pipeline_df = pd.read_excel(path_to_xlsx, sheet_name="tests_config")
    agg_config = pd.read_excel(path_to_xlsx, sheet_name="agg_config")

    code_columns = ["test_key", "import_path", "block_key", "params", "informative"]
    excel_columns = [
        "test_key",
        "Название блока",
        "Название",
        "Цель",
        "Интерпретация",
        "Границы красный",
        "Границы желтый",
        "Границы зеленый",
    ]

    pipeline_for_coding = (
        full_pipeline_df[code_columns].set_index("test_key").T.to_dict()
    )

    for test in pipeline_for_coding:
        pass
        pipeline_for_coding[test]["params"] = ast.literal_eval(
            pipeline_for_coding[test]["params"]
        )

    pipeline_for_excel = (
        full_pipeline_df[excel_columns].set_index("test_key").T.to_dict()
    )

    return pipeline_for_coding, pipeline_for_excel, agg_config
