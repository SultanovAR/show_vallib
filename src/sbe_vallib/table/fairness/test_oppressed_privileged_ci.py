from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sbe_vallib.utils.quantization import Quantizer
from sbe_vallib.utils.fsdict import FSDict
import sbe_vallib.utils.pd_np_interface as interface
from sbe_vallib.table.fairness.detect_oppr_priv import get_oppressed_privileged_mask
from sbe_vallib.utils.gini_ci import gini_conf_interval
from sbe_vallib.table.fairness.interval import Interval
from sbe_vallib.utils.report_helper import worst_semaphore


def color_criteria(oppr, priv, conf_lvls=(0.95, 0.99)):
    for group in (oppr, priv):
        for ci_lvl in conf_lvls:
            if isinstance(group[ci_lvl], str):
                return 'gray'
            group[ci_lvl] = Interval(*group[ci_lvl])

    color = 'red'
    if (not (oppr[max(conf_lvls)] & priv[max(conf_lvls)]).is_empty)\
            or ((oppr[max(conf_lvls)].left > 0.2) and (priv[max(conf_lvls)].left > 0.2)):
        color = 'yellow'
    if (not (oppr[min(conf_lvls)] & priv[min(conf_lvls)]).is_empty)\
            or ((oppr[max(conf_lvls)].left > 0.4) and (priv[max(conf_lvls)].left > 0.4)):
        color = 'green'
    if (oppr[min(conf_lvls)].length > 1) or (priv[min(conf_lvls)].length > 1):
        color = 'gray'
    return color


def report_oppressed_privileged_ci(ci_by_feature, repr_value_by_feature, conf_lvls, round_upto=3):
    df = defaultdict(list)
    df['Защищенная характеристика'] = list(
        ci_by_feature.keys())
    for feat in ci_by_feature.keys():
        oppr_ci = ci_by_feature[feat]['oppr']
        priv_ci = ci_by_feature[feat]['priv']
        for conf_lvl in conf_lvls:
            if isinstance(oppr_ci[conf_lvl], str):
                rounded_oppr = oppr_ci[conf_lvl]
            else:
                rounded_oppr = [round(i, round_upto)
                                for i in oppr_ci[conf_lvl]]

            if isinstance(priv_ci[conf_lvl], str):
                rounded_priv = priv_ci[conf_lvl]
            else:
                rounded_priv = [round(i, round_upto)
                                for i in priv_ci[conf_lvl]]

            df[f'{conf_lvl * 100}%-ый дов. интервал для угнетаемой группы'].append(
                rounded_oppr)
            df[f'{conf_lvl * 100}%-ый дов. интервал для привелегированной группы'].append(
                rounded_priv)
        df['Интервал значений признака угнетаемой группы'].append(
            repr_value_by_feature[feat]['oppr'])
        df['Интервал значений признака привилегированной группы'].append(
            repr_value_by_feature[feat]['priv'])
        df['Результат теста'].append(
            color_criteria(oppr_ci, priv_ci, conf_lvls))
    df = pd.DataFrame(df)

    result = {
        'semaphore': worst_semaphore(df['Результат теста']),
        'result_dict': None,
        'result_dataframes': [df],
        'result_plots': []
    }
    return result


def test_oppressed_privileged_ci(model, sampler, protected_feats,
                                 min_freq_value=0.01, min_amount_pos=50,
                                 conf_lvls=(0.95, 0.99), feature_preprocessor=None, precomputed: FSDict = None, **kwargs):
    """
    "Часть 31. Тест 7.4. Сравнение ранжирующей способности внутри защищенной
    характеристики". Test measures gini's confidence interval under oppressed
    and privileaged groups. After that it compares confidence intervals.

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
    confidence_levels : tuple, optional
        a tuple of confidence levels for whom confidence intervals
        will be computed, by default (0.95, 0.99)
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
        f'{conf_lvl * 100}%-ый доверительный интервал для угнетаемой группы'
        f'{conf_lvl * 100}%-ый доверительный интервал для привелегированной группы'
        'Результат теста'
        'Интервал значения признака угнетаемой группы'
        'Интервал значения признака привилегированной группы'
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
            min_freq_pos=min_amount_pos
        )
        if precomputed is not None:
            precomputed['mask_oppressed_privileged'] = mask_oppressed_privileged

    if 'y_pred' not in sampler.oos:
        sampler.reset()
        sampler.oos['y_pred'] = model.predict_proba(sampler.oos['X'])

    ci_by_feature = defaultdict(dict)
    repr_value_by_feature = defaultdict(dict)
    for protected_feat in mask_oppressed_privileged:
        for group_type in ('oppr', 'priv'):
            group_mask = mask_oppressed_privileged[protected_feat][group_type]['mask']
            group_target = interface.get_by_mask(
                sampler.oos['y_true'], group_mask)
            group_score = interface.get_by_mask(
                sampler.oos['y_pred'], group_mask)
            group_score = np.array(group_score)[:, 1]
            repr_value = mask_oppressed_privileged[protected_feat][group_type]['representative_value']

            ci_by_level = dict()
            for ci_lvl in conf_lvls:
                if (group_target.mean() == 0.0) or (group_target.mean() == 1.0):
                    ci = 'невозможно рассчитать, в группе один класс'
                else:
                    ci = gini_conf_interval(
                        group_score, group_target, alpha=ci_lvl)
                ci_by_level.update({ci_lvl: ci})
            ci_by_feature[protected_feat][group_type] = ci_by_level
            repr_value_by_feature[protected_feat][group_type] = repr_value

    result = report_oppressed_privileged_ci(
        ci_by_feature, repr_value_by_feature, conf_lvls)
    if precomputed is not None:
        # for test_delete_protected
        precomputed['result_oppressed_privileged_ci'] = result
    return result
