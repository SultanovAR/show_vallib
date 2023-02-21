from sbe_vallib.validation.table.general_tests.model_quality import test_ci

pipeline_31 = {
    "tests": {
        "test_1_1": {
            "block": "data_quality",
            "callable": test_ci,
            "params": {
                "n_iter": 200,
                "use_predict_proba": True,
            },
            "aggregation_most_important": True,
        }
    },
    "aggregation_mode": {},
    "aggregation_mode_by_block": {},
} 


def parse():
    pass