import os
import datetime
import shelve
import traceback

import pandas as pd

from sbe_vallib.parser import parse_pipeline, get_callable_from_path
from sbe_vallib.utils.fsdict import FSDict


class Validation:
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline="sbe_vallib/table/pipelines/Config_31.xlsx",
        exclude_tests=[],
        custom_tests={},
        store_path='./test_results'
    ):

        self.model = model
        self.sampler = sampler
        self.scorer = scorer
        self._store_path_prefix = store_path

        if isinstance(pipeline, str):
            tests, tests_desc, agg_config = self._parse_pipeline(pipeline)
        else:
            tests, tests_desc, agg_config = pipeline

        self.pipeline = {
            "tests": tests,
            "tests_desc": tests_desc,
            "aggregation_mode_by_block": agg_config,
        }

        self.pipeline["tests"].update(custom_tests)
        self.pipeline["tests"] = {
            key: self.pipeline["tests"][key]
            for key in self.pipeline["tests"]
            if key not in exclude_tests
        }

        self.result_of_validation = None

    def _create_store_dir(self):
        store_dir = os.path.join(
            self._store_path_prefix, datetime.datetime.now().strftime('%H_%M_%d,%m,%Y'))
        os.makedirs(store_dir, exist_ok=True)
        return store_dir

    def validate(self):
        tests_result = dict()
        store_dir = self._create_store_dir()
        precomputed = FSDict(os.path.join(
            store_dir, 'precomputes'), compress=0)
        for test_name in self.pipeline["tests"]:
            try:
                if "import_path" in self.pipeline["tests"][test_name]:
                    test_function = get_callable_from_path(
                        self.pipeline["tests"][test_name]["import_path"]
                    )
                else:
                    test_function = self.pipeline["tests"][test_name]["callable"]
                test_params = self.pipeline["tests"][test_name].get(
                    "params", {})
                tests_result[test_name] = test_function(
                    model=self.model,
                    sampler=self.sampler,
                    scorer=self.scorer,
                    precomputed=precomputed,
                    **test_params
                )
            except Exception as e:
                tests_result[test_name] = {
                    'semaphore': traceback.format_exc(),
                    'result_dict': None,
                    'result_dataframes': [],
                    'result_plots': []
                }
                print(tests_result[test_name]['semaphore'])

        result_of_validation = self.aggregate_results(tests_result)
        return result_of_validation

    def _parse_pipeline(self, path_to_pipeline):
        return parse_pipeline(path_to_pipeline)

    def aggregate_results(self, tests_result):
        return tests_result
