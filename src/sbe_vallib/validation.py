import os
import datetime
import traceback
import typing as tp

import pandas as pd

from sbe_vallib.parser import parse_pipeline, get_callable_from_path
from sbe_vallib.xlsx_aggregator import Aggregator
from sbe_vallib.utils.fsdict import FSDict


class Validation:
    def __init__(
            self,
            model,
            sampler,
            scorer,
            pipeline: tp.Union[os.PathLike, tp.Tuple[dict]
                               ] = "sbe_vallib/table/pipelines/Config_31.xlsx",
            tests_params: dict = {},
            exclude_tests: tp.List[str] = [],
            custom_tests: tp.Dict = {},
            store_path: os.PathLike = './test_results'):
        """
        Validation runs all tests from 'pipeline' and then aggregates results into excel file.

        :param model: the model to be tested, you can find discription in README.md file
        :param sampler: Generate a sample of the data, you can find discription in README.md file
        :param scorer: Evaluate the model, you can find discription in README.md file
        :param pipeline: path to excel file or tp.Tuple[dict] object. About excel file you can read in README.md
            The format of the tp.Tuple[dict] object is following: (test_desc, agg_config) where
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
        :param exclude_tests: Exclude tests from the pipeline
        :param custom_tests: Add custom tests to the pipeline. It should be dict with the following format:
            {
                'test_key': {
                    'callable': tp.Callable,
                    'block_key': tp.Optional[str],
                    'informative': tp.Optional[bool],
                    'params': {
                        'param_1': param_value_1,
                        ...
                    }
                }
            }
        :param store_path: Specify the path where the results of validation will be stored
        :return: a dict with the following format:
            {
                'test_key': {
                    "semaphore": str one of {"gray", "green", "yellow", "red"},
                    "result_dict": python object,
                    "result_dataframes": List[pd.DataFrame],
                    "result_plots": List[PIL.Image],
                }
            }
        """

        self.model = model
        self.sampler = sampler
        self.scorer = scorer
        self.tests_params = tests_params
        self._store_path_prefix = store_path

        if isinstance(pipeline, str):
            tests_desc, agg_config = self._parse_pipeline(pipeline)
        else:
            tests_desc, agg_config = pipeline

        self.pipeline = {
            "tests_desc": tests_desc,
            "aggregation_mode_by_block": agg_config,
        }

        self.pipeline["tests_desc"].update(custom_tests)
        self.pipeline["tests_desc"] = {
            key: self.pipeline["tests_desc"][key]
            for key in self.pipeline["tests_desc"]
            if key not in exclude_tests
        }

        self.result_of_validation = None

    def _create_store_dir(self):
        store_dir = os.path.join(
            self._store_path_prefix, datetime.datetime.now().strftime('%H_%M_%S_%d,%m,%Y'))
        os.makedirs(store_dir, exist_ok=True)
        return store_dir

    def validate(self, save_excel=True):
        tests_result = dict()
        store_dir = self._create_store_dir()
        precomputed = FSDict(os.path.join(
            store_dir, 'precomputes'), compress=0)
        for test_name in self.pipeline["tests_desc"]:
            print(f'Test: {test_name} started')
            try:
                if "import_path" in self.pipeline["tests_desc"][test_name]:
                    test_function = get_callable_from_path(
                        self.pipeline["tests_desc"][test_name]["import_path"]
                    )
                else:
                    test_function = self.pipeline["tests_desc"][test_name]["callable"]
                test_params = self.pipeline["tests_desc"][test_name].get(
                    "params", {})
                test_params.update(self.tests_params)
                tests_result[test_name] = test_function(
                    model=self.model,
                    sampler=self.sampler,
                    scorer=self.scorer,
                    precomputed=precomputed,
                    **test_params
                )
            except Exception:
                text_error = traceback.format_exc()
                print(f'\tTest: {test_name} raised an exception:')
                print(text_error)
                tests_result[test_name] = {
                    'semaphore': 'gray',
                    'result_dict': None,
                    'result_dataframes': [pd.DataFrame([{'error': text_error}])],
                    'result_plots': []
                }

        if save_excel:
            aggregator = Aggregator(
                save_path=os.path.join(store_dir, 'excel_report.xlsx'))
            aggregator.agregate(
                tests_result, self.pipeline['tests_desc'], self.pipeline['aggregation_mode_by_block'])
        return tests_result

    def _parse_pipeline(self, path_to_pipeline):
        return parse_pipeline(path_to_pipeline)
