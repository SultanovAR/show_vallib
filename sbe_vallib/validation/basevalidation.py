# from abc import abstractmethod

# from typing import List

# import os
# from utility import (
#     param_types,
#     printcolor,
#     semafore_agregation,
#     agg_results,
#     parse_params,
#     mincolor,
# )

from tqdm import tqdm

class BaseValidation:
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline='31',
        # exclude_tests=[],
        custom_tests=[],
    ):

        self.model = model
        self.sampler = sampler
        self.scorer = scorer
        
        # self.exclude_tests = exclude_tests,
        self.custom_tests = custom_tests,
        
        self.default_test_list = []
        self.test_list = []
        self.result_of_validation = None

    def validate(self):
        tests_result = dict()
        for test in tqdm(self.test_list):
            tests_result[test.__name__] = test(self.model, self.sampler, self.scorer)

        self.result_of_validation = tests_result
        
        return tests_result
    
    def aggregate_results(self):
        pass
    
# [
#     {'test_key': 'test_1_1',
#      'params': {
#          'cutoff',
#     'test_func' : callable
#      }}
# ]