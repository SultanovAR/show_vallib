import typing as tp

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


def std_gini(auc_value: float, target: tp.List):
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value ** 2 / (1 + auc_value)
    n_A = np.sum(target)
    n_N = len(target) - n_A

    sigma_auc = np.sqrt((auc_value * (1 - auc_value)
                         + (n_A - 1) * (q1 - auc_value ** 2)
                         + (n_N - 1) * (q2 - auc_value ** 2))
                        / (n_A * n_N))
    sigma_gini = 2 * sigma_auc
    return sigma_gini


def gini_conf_interval(scores: tp.List, target: tp.List, alpha: float = 0.95):
    """
    https://stats.stackexchange.com/questions/296748/how-to-get-approximative-confidence-interval-for-gini-and-auc
    """
    auc_value = roc_auc_score(y_score=scores, y_true=target)
    gini_value = -1 + 2 * auc_value
    std = std_gini(auc_value, target)
    z_coef = abs(stats.norm.ppf((1 - alpha) / 2))
    return gini_value - z_coef * std, gini_value + z_coef * std
