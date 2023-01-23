from sbe_vallib.validation.basevalidation import BaseValidation
from sbe_vallib.validation.typedata.table.data_quality import (
    train_test_independence_test,
)


class TableValidation(BaseValidation):
    def __init__(
        self,
        model,
        sampler,
        scorer,
        include_tests=[],
        exclude_tests=[],
        custom_tests=[],
        **kwargs
    ):
        super().__init__(
            model, sampler, scorer, include_tests, exclude_tests, custom_tests
        )

        self.config = kwargs

        self.standart_params = {
            "stratify": None,
            "stratify_by_cols": None,
            "n_iter_conf": None,
            "bootstrap_conf_int": None,
            "n_iter_shuffle": None,
            "n_iter_split": None,
            "n_iter_indep": None,
            "recalc_conf_int": None,
            "recalc_feat_imp": None,
            "recalc_split_indep": None,
            "exclude_list": None,
            "include_list": None,
            "verbose": None,
        }

        # define a common set of tests for tabular data
        self.default_test_list = [train_test_independence_test]

    def check_validation_settings(self):
        pass
