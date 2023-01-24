from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


def gini_score(y_true, y_proba):
    return 2 * roc_auc_score(y_true, y_proba) - 1


BINARY_METRICS = {
    "accuracy_score": {"callable": accuracy_score, "use_probas": False},
    "precision_score": {"callable": precision_score, "use_probas": False},
    "recall_score": {"callable": recall_score, "use_probas": False},
    "f1_score": {"callable": f1_score, "use_probas": False},
    "gini": {"callable": gini_score, "use_probas": True},
}

REGRESSION_METRICS = {
    "MSE": {"callable": mean_squared_error},
    "MAE": {"callable": mean_absolute_error},
    "MAPE": {"callable": mean_absolute_percentage_error},
    "R2": {"callable": r2_score},
}

MULTICLASS_METRICS = {
    "roc_auc_score": {
        "callable": roc_auc_score,
        "multiclass": "ovr",
        "average": "micro",
    },
    "f1_score": {"callable": f1_score, "multiclass": None, "average": "micro"},
    "recall_score": {"callable": recall_score, "multiclass": None, "average": "micro"},
    "precision_score": {
        "callable": precision_score,
        "multiclass": None,
        "average": "micro",
    },
} # metric params
