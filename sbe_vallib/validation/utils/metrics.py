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

import seqeval.metrics
from seqeval.scheme import IOB2


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
}  # metric params


def ner_recall_score(y_true, y_pred, **kwargs):
    cr = seqeval.metrics.classification_report(
        y_true, y_pred, output_dict=True)
    return {tag.replace(' avg', ''): cr[tag]['recall'] for tag in cr}


def ner_f1_score(y_true, y_pred, **kwargs):
    cr = seqeval.metrics.classification_report(
        y_true, y_pred, output_dict=True)
    return {tag.replace(' avg', ''): cr[tag]['f1-score'] for tag in cr}


def ner_precision_score(y_true, y_pred, **kwargs):
    cr = seqeval.metrics.classification_report(
        y_true, y_pred, output_dict=True)
    return {tag.replace(' avg', ''): cr[tag]['precision'] for tag in cr}


def ner_support_score(y_true, y_pred, **kwargs):
    cr = seqeval.metrics.classification_report(
        y_true, y_pred, output_dict=True)
    return {tag.replace(' avg', ''): cr[tag]['support'] for tag in cr}


NER_IOB_METRICS = {
    "precision_score": {"callable": ner_precision_score, "params": {"schema": IOB2}},
    "f1_score": {"callable": ner_f1_score, "params": {"schema": IOB2}},
    "recall_score": {"callable": ner_recall_score, "params": {"schema": IOB2}},
    "support": {"callable": ner_support_score, "params": {"schema": IOB2}},
}
