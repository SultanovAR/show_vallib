import numpy as np
import pandas as pd
import math
import typing as tp

from scipy.special import rel_entr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sbe_vallib.sampler import SupervisedSampler
from sbe_vallib.utils import pd_np_interface as interface
from sbe_vallib.utils.quantization import Quantization
from sbe_vallib.utils.report_helper import semaphore_by_threshold, worst_semaphore


def psi(train, oos, base=None, axis=0):
    EPS = 1e-7
    train = (train + EPS) / np.sum(train + EPS)
    oos = (oos + EPS) / np.sum(oos + EPS)
    return np.sum(rel_entr(train, oos) + rel_entr(oos, train))


def report_factor_psi(psi_of_feat: tp.Dict, threshold: tp.Tuple):
    """Create dict with the SberDS format

    Parameters
    ----------
    psi_of_feat : tp.Dict
        dictionary with the following format {'feature': psi_value}
    """

    semaphores = {feat: semaphore_by_threshold(
        psi_of_feat[feat]['psi'], threshold, False) for feat in psi_of_feat}

    res_df = psi_of_feat
    for feat in psi_of_feat:
        res_df[feat]['feature'] = feat
        res_df[feat]['hist_train'] = " ;".join([
            f'{i:.3f}' for i in psi_of_feat[feat]['hist_train']])
        res_df[feat]['hist_oos'] = " ;".join([
            f'{i:.3f}' for i in psi_of_feat[feat]['hist_oos']])
        res_df[feat]['semaphore'] = semaphores[feat]

    result = dict()
    result['semaphore'] = worst_semaphore(semaphores.values())
    result['result_dict'] = psi_of_feat
    result['result_dataframes'] = [pd.DataFrame.from_records(res_df)]
    result['result_plots'] = []
    return result


def get_feature_types(data, discr_uniq_val=10, discr_val_share=0.8):
    """
    Determine type of every DataFrame column: discrete or continuous.
    If the number of observations with discr_uniq_val most common unique values
        is more than discr_val_share of total sample, column is considered discrete.

    :param df: Input DataFrame
    :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
    :param discr_val_share: share of most common values that is enough to define factor as discrete
    :return: dict of {'feature': 'Discrete' or 'Continuous'}
    """
    data_dropna = np.array(data)[~np.isnan(data)]
    # Get sorted values by counts descending
    unique_counts = np.unique(data_dropna, return_counts=True)[1]
    unique_counts = sorted(unique_counts, reverse=True)

    if np.sum(unique_counts[:discr_uniq_val]) / len(data_dropna) >= discr_val_share:
        return 'Discrete'
    return 'Continuous'


def test_factor_psi(sampler,
                    merge_upto_quantile: float = 0.05,
                    rounding_precision_bins: int = 5,
                    discr_uniq_val: int = 10,
                    discr_val_share: float = 0.8,
                    threshold: tp.Tuple = (0.2, 10**10), **kwargs):
    sampler.reset()
    x_train, x_oos = sampler.train['X'], sampler.oos['X']

    encoder = LabelEncoder()
    quantizer = Quantization(merge_upto_quantile, rounding_precision_bins)

    psi_of_feat = {}
    for col in interface.all_columns(x_oos):
        feat_type = get_feature_types(interface.get_columns(x_train, col),
                                      discr_uniq_val, discr_val_share)
        train_col = np.array(interface.get_columns(x_train, col))[:, None]
        oos_col = np.array(interface.get_columns(x_oos, col))[:, None]
        if feat_type == 'Discrete':
            train_col = encoder.fit_transform(train_col)
            oos_col = encoder.transform(oos_col)
        bins = quantizer.fit(train_col).bins[0]

        hist_train_col, _ = np.histogram(train_col, bins=bins)
        hist_oos_col, _ = np.histogram(oos_col, bins=bins)
        psi_of_feat[col] = {
            'psi': psi(hist_train_col, hist_oos_col),
            'feat_type': feat_type,
            'bin_count': len(bins),
            'hist_train': hist_train_col,
            'hist_oos': hist_oos_col,
            'bins': bins
        }

    return report_factor_psi(psi_of_feat, threshold)
