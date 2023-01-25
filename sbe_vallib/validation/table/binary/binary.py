from sbe_vallib.validation.table.tablevalidation import TableValidation  # , pipeline_31
from sbe_vallib.validation.scorer import BinaryScorer
from sbe_vallib.validation.table.general_tests.model_quality import key_metric_test


class BinaryValidation(TableValidation):
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline="pipeline_31",
        exclude_tests=[],
        custom_tests=[],
        **kwargs,
    ):
        super().__init__(
            model, sampler, scorer, pipeline, exclude_tests, custom_tests
        )

    def test_1_2(self):
        pass
