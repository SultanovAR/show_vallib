from sbe_vallib.validation.basevalidation import BaseValidation
from sbe_vallib.validation.table.general_tests.data_quality import (
    train_test_independence_test,
)


class TableValidation(BaseValidation):
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline,
        exclude_tests=[],
        custom_tests=[],
        **kwargs,
    ):
        super().__init__(
            model, sampler, scorer, pipeline, exclude_tests, custom_tests
        )
        
        self.default_test_list = self._parse_pipeline()
        self.config = kwargs
        # define a common set of tests for tabular data
