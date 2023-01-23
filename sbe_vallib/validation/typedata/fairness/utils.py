from typing import List, Union

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from scipy.stats import norm


def bootstraped_score(y_true, y_pred, score_func, score_func_params={}, n_bootstrap=100, random_seed=0):
    assert len(y_pred) == len(y_true), 'lenghts of y_pred and y_true should be the same'
    np.random.seed(random_seed)
    scores = []
    for _ in range(n_bootstrap):
        indxs = np.random.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        scores.append(score_func(y_true[indxs], y_pred[indxs], **score_func_params))
    return scores
        

def copy_docstring_from(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper


@copy_docstring_from(roc_auc_score)
def gini(*args, **argv):
    return -1 + 2 * roc_auc_score(*args, **argv)


def std_gini(auc_value: float, target: List):
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value ** 2 / (1 + auc_value)
    n_A = np.sum(target)
    n_N = len(target) - n_A
    
    sigma_auc = np.sqrt((auc_value * (1 - auc_value)\
                         + (n_A - 1) * (q1 - auc_value ** 2)\
                         + (n_N - 1) * (q2 - auc_value ** 2))\
                        / (n_A * n_N))
    sigma_gini = 2 * sigma_auc
    return sigma_gini


def gini_conf_interval(scores: List, target: List, alpha: float=0.95):
    """
    https://stats.stackexchange.com/questions/296748/how-to-get-approximative-confidence-interval-for-gini-and-auc
    """
    auc_value = roc_auc_score(y_score = scores, y_true = target)
    gini_value = -1 + 2 * auc_value
    std = std_gini(auc_value, target)
    z_coef = abs(norm.ppf((1 - alpha) / 2))
    return gini_value - z_coef * std, gini_value + z_coef * std


def shuffle_data(*args: List[pd.DataFrame], random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    idxs = np.random.permutation(len(args[0]))
    return [arg.iloc[idxs] for arg in args]


def get_columns(data: Union[np.ndarray, pd.DataFrame], columns):
    if isinstance(data, np.ndarray):
        return data[:, columns]
    if isinstance(data, pd.DataFrame):
        return data[columns]

    
def set_column(where: Union[np.ndarray, pd.DataFrame],
                data: Union[np.ndarray, pd.Series],
                column):
    if isinstance(where, np.ndarray):
        where[:, column] = data
    if isinstance(where, pd.DataFrame):
        where.loc[:, column] = data

        
def all_columns(data: Union[np.ndarray, pd.DataFrame]):
    if isinstance(data, np.ndarray):
        return np.arange(data.shape[1]).astype(int)
    if isinstance(data, pd.DataFrame):
        return data.columns



