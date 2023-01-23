from sbe_vallib.validation.table.tablevalidation import TableValidation
from sbe_vallib.validation.table.general_tests.model_quality import key_metric_test


class BinaryValidation(TableValidation):
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline='31',
        exclude_tests=[],
        custom_tests=[],
        **kwargs,
    ):
        super().__init__(
            model, sampler, scorer, pipeline, exclude_tests, custom_tests
        )

        self.config.update(kwargs)
        self.test_list.extend([key_metric_test, self.test_1_2])

    def test_1_2(self):
        pass
