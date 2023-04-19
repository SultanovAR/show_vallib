from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

from sbe_vallib.table.fairness.detect_oppr_priv import get_oppressed_privileged_mask
from sbe_vallib.utils.report_helper import worst_semaphore
from sbe_vallib.utils.quantization import Quantizer
from sbe_vallib.table.fairness.swapping_oppr_priv import get_swapped_oppr_priv_predictions
from sbe_vallib.utils.cat_features import get_cat_features


def report_tprd_fprd_delta(tprd_fprd_delta_by_feat: dict, repr_value_by_feat, thresholds=(0.1, 0.2)):
    """
    Creates a report for the results of 'test_tprd_fprd_delta' in pandas DataFrame format

    Parameters
    ----------
    tprd_fprd_delta_by_feat: Dict[Dict[str]]
        Dict with the format: {"feature": {"tprd": value, "fprd": value }}
    thresholds: Tuple
        Tuple with the format: (yellow_threshold, red_threshold)

    Returns
    -------
    pd.DataFrame with columns:
        'Защищенная характеристика'
        'Изменение TPRD после перестановки значений'
        'Изменение FPRD после перестановки значений'
        'Результат теста'
    """
    def color_criteria(tprd, fprd):
        diff = max(tprd, fprd)
        if diff <= min(thresholds):
            return 'green'
        elif min(thresholds) < diff <= max(thresholds):
            return 'yellow'
        return 'red'

    df = pd.DataFrame()
    df['Защищенная характеристика'] = list(
        tprd_fprd_delta_by_feat.keys())
    df['Абс. изменение TPRD после перестановки значений'] = [
        i['tprd'] for i in tprd_fprd_delta_by_feat.values()]
    df['Абс. изменение FPRD после перестановки значений'] = [
        i['fprd'] for i in tprd_fprd_delta_by_feat.values()]
    df[f'Интервал значений признака угнетаемой группы'] = [
        repr_value_by_feat[feat]['oppr'] for feat in repr_value_by_feat]
    df[f'Интервал значений признака привилегированной группы'] = [
        repr_value_by_feat[feat]['priv'] for feat in repr_value_by_feat]
    df['Результат теста'] = [
        color_criteria(i['tprd'], i['fprd']) for i in tprd_fprd_delta_by_feat.values()]
    result = {
        'result_dataframes': [df],
        'result_plots': [],
        'result_dict': None,
        'semaphore': worst_semaphore(df['Результат теста'])
    }
    return result


def test_tprd_fprd_delta(sampler, model,
                         protected_feats: list,
                         min_freq_value=0.01,
                         min_freq_pos=0.05,
                         thresholds=(0.1, 0.2),
                         feature_preprocessor=None,
                         cat_features=None,
                         precomputed=None, **kwargs):
    """
    "Часть 31. Тест 7.3. Анализ динамики TPR и FPR при изменении защищенной характеристики".
    Test measures True-Positive-Rate-Difference and False-Positive-Rate-Difference
    when feature values for oppressed and privileaged groups are swapped.

    Returns
    -------
    pd.DataFrame with columns:
        'Защищенная характеристика'
        'Изменение TPRD после перестановки значений'
        'Изменение FPRD после перестановки значений'
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
    tprd_fprd_delta_by_feat = defaultdict(dict)
    repr_value_by_feat = defaultdict(dict)
    for feat in swapped_oppr_priv_predictions:
        groups_metric = {'tpr': [], 'fpr': [], 'swap_tpr': [], 'swap_fpr': []}
        for group_type in ('oppr', 'priv'):
            source_preds = np.array(
                swapped_oppr_priv_predictions[feat][group_type]['source_preds'] > cutoff, dtype=int)
            swap_preds = np.array(
                swapped_oppr_priv_predictions[feat][group_type]['swapped_preds'] > cutoff, dtype=int)
            target = np.array(
                swapped_oppr_priv_predictions[feat][group_type]['target'], dtype=int)

            groups_metric['tpr'].append(source_preds[target == 1].mean())
            groups_metric['fpr'].append(source_preds[target == 0].mean())
            groups_metric['swap_tpr'].append(
                swap_preds[target == 1].mean())
            groups_metric['swap_fpr'].append(
                swap_preds[target == 0].mean())
            repr_value_by_feat[feat][group_type] = mask_oppressed_privileged[feat][group_type]['representative_value']

        diff_metric = dict()
        for metric in groups_metric:
            diff_metric[metric] = max(
                groups_metric[metric]) - min(groups_metric[metric])

        tprd_fprd_delta_by_feat[feat].update({
            'tprd': abs(diff_metric['swap_tpr'] - diff_metric['tpr']),
            'fprd': abs(diff_metric['swap_fpr'] - diff_metric['fpr'])
        })

    result = report_tprd_fprd_delta(
        tprd_fprd_delta_by_feat, repr_value_by_feat, thresholds)
    if precomputed is not None:
        # for test_delete_protected
        precomputed['result_tprd_fprd_delta'] = result
    return result
