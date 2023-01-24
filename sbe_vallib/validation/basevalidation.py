import pandas as pd
from sbe_vallib.validation.basevalidation import BaseValidation
from sbe_vallib.validation.table.general_tests.data_quality import (
    train_test_independence_test,
)
from sbe_vallib.validation.table.general_tests.model_quality import test_ci


pipeline_31 = {
        'tests': {
            'test_1_1': {
                    'block': 'data_quality',
                    'callable': test_ci,
                    'params': {
                        "n_iter": 200,
                        "use_predict_proba": True,
                    },
                    "aggregation_most_important": True
                }
            },
        'aggregation_mode': {},
        'aggregation_mode_by_block': {}
    }

def custom_test(model, scorer, sampler, **kwargs):
    return {
        'semaphore': 'grey',
        'result_dict': {},
        'result_dataframes': [pd.DataFrame()],
        'result_plots': []
        } # сделать один в один со SberDS

custom_tests = {
        'test_0_0': {
            'block': 'model_stability',
            'callable': custom_test,
            'params': {}
        },
        'test_0_0_1': {
            'block': 'model_stability',
            'callable': custom_test,
            'params': {'a': 'b'}
        }
    }


class BaseValidation:
    def __init__( # поправить
        self,
        model,
        sampler,
        scorer,
        pipeline=pipeline_31,
        exclude_tests=[],
        custom_tests={},
    ):

        self.model = model
        self.sampler = sampler
        self.scorer = scorer
        
        if isinstance(pipeline, str):
            self.pipeline = self._parse_pipeline()
        else:
            self.pipeline = pipeline
            
        self.pipeline['tests'].update(custom_tests)
        self.pipeline['tests'] = {self.pipeline['tests'][key] for key in self.pipeline['tests'] if key not in exclude_tests}
        self.result_of_validation = None

    def validate(self):
        tests_result = dict()
        for test_name in (self.pipeline['tests']):
            test_function = self.pipeline['tests'][test_name]['callable']
            test_params = self.pipeline['tests'][test_name].get('params', {})
            tests_result[test_name] = test_function(model=self.model, sampler=self.sampler, scorer=self.scorer, **test_params)

        self.result_of_validation = tests_result
        
        return tests_result
    
    def _parse_pipeline(self):
        return dict()
    
    def aggregate_results(self):
        pass