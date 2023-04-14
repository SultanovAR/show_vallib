import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd

import sbe_vallib.utils.pd_np_interface as interface


def _get_value_range(series, mask, round_upto=4):
    data = interface.get_by_mask(series, mask)
    min_val = np.round(np.min(data), round_upto)
    max_val = np.round(np.max(data), round_upto)
    return f"[{min_val}-{max_val}]"


def get_oppressed_privileged_mask(x_train: pd.DataFrame,
                                  x_test: pd.DataFrame,
                                  y_test: tp.Union[np.ndarray, pd.Series],
                                  protected_feats: tp.List[str],
                                  feature_preprocessor,
                                  min_freq_value=0.05,
                                  min_freq_pos=0.00):
    """Defines oppressed and privileaged groups for each feature in protected_feats.
    an oppressed group - value of feature for which the target rate is minimal.
    a privileaged group - value of feature for which the target rate is maximal.
    Values of feature is grouping with quantizer

    Parameters
    ----------
    x_train : pd.DataFrame
        features from the train dataset
    x_test : pd.DataFrame
        features from the test dataset
    y_test : Union[np.ndarray, pd.Series]
        target from the test dataset
    protected_feats : List[str]
        features for which discrimination tests will be provided
    feat_preproc : _type_
        preprocessor in sklearn format (with fit() and transform() methods)
        it discretizises/(imputes NaNs) features that are continues
        or have a lot of unique values.
    min_freq_value : float, optional
        a value can represent a privileged or oppressed group only if
        freq(value) > min_freq_value, by default 0.01
    min_amount_pos : int, optional
        a value can represent a privileged or oppressed group only if
        target values has enough positive labels, by default 50

    Returns
    -------
    Dict with the following format
    {
        'feature': {
            'oppr': {'mask': array, 'representative_value': str, 'value': value},
            'priv': {'mask': array, 'representative_value': str, 'value': value}
        }
    }
    """
    result = defaultdict(dict)
    x_test_processed = feature_preprocessor.fit(x_train).transform(x_test)
    x_test_processed = pd.DataFrame(
        x_test_processed, columns=interface.all_columns(x_train))
    for feat in protected_feats:
        feat_target = pd.DataFrame({
            feat: x_test_processed[feat].values,
            'target': np.array(y_test)})

        # filtering
        num_positive = feat_target.groupby(feat)['target'].mean()
        num_values = feat_target[feat].nunique()
        mask = np.ones(len(feat_target), dtype=bool)
        if num_values > 2:
            counts = feat_target[feat].value_counts()
            enough_samples = (counts > min_freq_value * len(feat_target))
            enough_positive = (num_positive > min_freq_pos * len(feat_target))
            well_presented_values = (enough_samples & enough_positive)
            well_presented_values = well_presented_values[well_presented_values]
            mask &= feat_target[feat].isin(well_presented_values.index)

        # oppressed and privileaged
        grouped_target = feat_target.loc[mask].groupby(feat)['target'].mean()
        oppr_priv_info = {
            'oppr': {'value': grouped_target.idxmin()},
            'priv': {'value': grouped_target.idxmax()}
        }
        for group_type in oppr_priv_info:
            value_mask = feat_target[feat] == oppr_priv_info[group_type]['value']
            feat_mask = (mask & value_mask).values
            oppr_priv_info[group_type]['mask'] = feat_mask
            oppr_priv_info[group_type]['representative_value'] = _get_value_range(
                x_test[feat], np.array(feat_mask))
        result[feat] = oppr_priv_info
    return result
