from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sbe_vallib.table.fairness.detect_oppr_priv import get_oppressed_privileged_mask
from sbe_vallib.utils.report_helper import worst_semaphore
from sbe_vallib.utils.quantization import Quantizer
from sbe_vallib.table.fairness.swapping_oppr_priv import get_swapped_oppr_priv_predictions
from sbe_vallib.utils.cat_features import get_cat_features


def report_target_rate_delta(tr_delta_by_feat: dict, repr_value_by_feat):
    """
    Creates a report for the results of 'test_target_rate_delta' in pandas DataFrame format

    Parameters
    ----------
    tr_delta_by_feat: Dict
        Dict with the format: {"feature": {"oppr": value, "priv": value}}

    Returns
    -------
    pd.DataFrame with columns:
        'Защищенная характеристика'
        'Относительное изменение частоты таргета у угнетаемой группы'
        'Относительное изменение частоты таргета у привилигированной группы'
        'Результат теста'
    """
    def color_criteria(tr_delta_first, tr_delta_second):
        if max(abs(tr_delta_first), abs(tr_delta_second)) > 0.3:
            if tr_delta_first * tr_delta_first < 0:
                return 'red'
            else:
                return 'yellow'
        return 'green'

    df = pd.DataFrame()
    df['Защищенная характеристика'] = list(tr_delta_by_feat.keys())
    df['Отн. изменение частоты таргета у угнетаемой группы'] = [
        i['oppr'] for i in tr_delta_by_feat.values()]
    df['Отн. изменение частоты таргета у привилигированной группы'] = [
        i['priv'] for i in tr_delta_by_feat.values()]
    df[f'Интервал значений признака угнетаемой группы'] = [
        repr_value_by_feat[feat]['oppr'] for feat in repr_value_by_feat]
    df[f'Интервал значений признака привилегированной группы'] = [
        repr_value_by_feat[feat]['priv'] for feat in repr_value_by_feat]
    df['Результат теста'] = [
        color_criteria(i['oppr'], i['priv']) for i in tr_delta_by_feat.values()]

    result = {
        'semaphore': worst_semaphore(df['Результат теста']),
        'result_dataframes': [df],
        'result_plots': [],
        'result_dict': None
    }
    return result


def test_target_rate_delta(sampler, model,
                           protected_feats: list,
                           min_freq_value=0.01,
                           min_freq_pos=0.05,
                           feature_preprocessor=None,
                           cat_features=None,
                           precomputed=None, **kwargs):
    """
    "Часть 31. Тест 7.2. Сравнение прогнозного уровня целевой переменной
    при изменении защищенной характеристики". Test measures Target-Rate in predictions
    when feature values for oppressed and privileaged groups are swapped.

    Parameters
    ----------
    model ...
    sampler ...
    precomputed ...
    protected_feats : List[str]
        features for which discrimination tests will be provided
    feature_preprocessor : _type_
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
    pd.DataFrame with columns:
        'Защищенная характеристика'
        'Относительное изменение частоты таргета у угнетаемой группы'
        'Относительное изменение частоты таргета у привилигированной группы'
        'Результат теста'
    """
    assert len(np.unique(
        sampler.oos['y_true'])) == 2, 'the test was designed only for binary models'

    if feature_preprocessor is None:
        feature_preprocessor = Pipeline(steps=(
            ('imputer', SimpleImputer(strategy='constant', fill_value=np.inf)),
            ('quantizer', Quantizer())
        ))

    if (precomputed is not None) and ('mask_oppressed_privileged' in precomputed):
        mask_oppressed_privileged = precomputed['mask_oppressed_privileged']
    else:
        mask_oppressed_privileged = get_oppressed_privileged_mask(
            x_train=sampler.train['X'], x_test=sampler.oos['X'],
            y_test=sampler.oos['y_true'], protected_feats=protected_feats,
            feature_preprocessor=feature_preprocessor,
            min_freq_value=min_freq_value,
            min_freq_pos=min_freq_pos
        )
        if precomputed is not None:
            precomputed['mask_oppressed_privileged'] = mask_oppressed_privileged

    if (precomputed is not None) and ('swapped_oppr_priv_predictions' in precomputed):
        swapped_oppr_priv_predictions = precomputed['swapped_oppr_priv_predictions']
    else:
        if cat_features is None:
            cat_features = get_cat_features(sampler.oos['X'])
        swapped_oppr_priv_predictions = get_swapped_oppr_priv_predictions(
            mask_oppressed_privileged, model, sampler.oos['X'],
            sampler.oos['y_true'], cat_features
        )
        if precomputed is not None:
            precomputed['swapped_oppr_priv_predictions'] = swapped_oppr_priv_predictions

    cutoff = sampler.train['y_true'].mean()
    target_rates_delta_by_feat = defaultdict(dict)
    repr_value_by_feat = defaultdict(dict)
    for feat in swapped_oppr_priv_predictions:
        for group_type in swapped_oppr_priv_predictions[feat]:
            source_preds = np.array(
                swapped_oppr_priv_predictions[feat][group_type]['source_preds'] > cutoff, dtype=int)
            swap_preds = np.array(
                swapped_oppr_priv_predictions[feat][group_type]['swapped_preds'] > cutoff, dtype=int)
            tr_delta = (swap_preds.mean() - source_preds.mean())\
                / (source_preds.mean() + 1e-10)
            target_rates_delta_by_feat[feat][group_type] = tr_delta
            repr_value_by_feat[feat][group_type] = mask_oppressed_privileged[feat][group_type]['representative_value']

    result = report_target_rate_delta(
        target_rates_delta_by_feat, repr_value_by_feat)
    if precomputed is not None:
        # for test_delete_protected
        precomputed['result_target_rate_delta'] = result
    return result
