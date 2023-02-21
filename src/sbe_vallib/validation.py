import pandas as pd
from sbe_vallib.parser import parse_pipeline, get_callable_from_path


class Validation:
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline="sbe_vallib/table/pipelines/Config_31.xlsx",
        exclude_tests=[],
        custom_tests={},
    ):

        self.model = model
        self.sampler = sampler
        self.scorer = scorer

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

    def validate(self):
        tests_result = dict()
        for test_name in self.pipeline["tests"]:
            if "import_path" in self.pipeline["tests"][test_name]:
                test_function = get_callable_from_path(
                    self.pipeline["tests"][test_name]["import_path"]
                )
            else:
                test_function = self.pipeline["tests"][test_name]["callable"]
            test_params = self.pipeline["tests"][test_name].get("params", {})
            tests_result[test_name] = test_function(
                model=self.model,
                sampler=self.sampler,
                scorer=self.scorer,
                **test_params
            )

        self.result_of_validation = tests_result

        return tests_result

    def _parse_pipeline(self, path_to_pipeline):
        return parse_pipeline(path_to_pipeline)

    def aggregate_results(self):
        pass
