from tqdm import tqdm

class BaseValidation:
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline='31',
        exclude_tests=[],
        custom_tests=[],
    ):

        self.model = model
        self.sampler = sampler
        self.scorer = scorer
        self.pipeline = pipeline
        
        self.custom_tests = custom_tests
        self.exclude_tests = exclude_tests
        self.default_test_list = [] # определяется через pipeline
        self.test_list = [] # определяется в соответствии с exclude и custom tests
        self.result_of_validation = None

    def validate(self):
        tests_result = dict()
        for test in tqdm(self.test_list):
            tests_result[test.__name__] = test(self.model, self.sampler, self.scorer)

        self.result_of_validation = tests_result
        
        return tests_result
    
    def _parse_pipeline(self):
        return dict()
    
    def aggregate_results(self):
        pass
    
# [
#     {'test_key': 'test_1_1',
#      'params': {
#          'cutoff',
#     'test_func' : callable
#      }}
# ]