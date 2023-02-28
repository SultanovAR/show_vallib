import typing as tp

import pandas as pd

from sbe_vallib.table.model_quality.test_key_metric import get_source_metrics
from sbe_vallib.utils.report_sberds import semaphore_by_threshold, best_semaphore


def report_test_key_metric_stability(metrics_oos, metrics_oot,
                                     metric_name: str,
                                     abs_thresholds: tp.Tuple,
                                     rel_thresholds: tp.Tuple,
                                     greater_is_better: bool = True):
    sign = 1 if greater_is_better else -1
    abs_reduction = sign * \
        (metrics_oos[metric_name] - metrics_oot[metric_name])
    rel_reduction = abs_reduction / metrics_oos[metric_name]

    abs_semaphore = semaphore_by_threshold(
        abs_reduction, abs_thresholds, greater_is_better=False)
    rel_semaphore = semaphore_by_threshold(
        rel_reduction, rel_thresholds, greater_is_better=False)
    semaphore = best_semaphore([abs_semaphore, rel_semaphore])

    df_res = pd.DataFrame([{
        f'Значение {metric_name} на выборке out-of-sample': metrics_oos[metric_name],
        f'Значение {metric_name} на выборке out-of-time': metrics_oot[metric_name],
        f'Изменение {metric_name}, абс-ое': abs_reduction,
        f'Изменение {metric_name}, отн-он': rel_reduction,
    }])

    return {
        'semaphore': semaphore,
        'result_dict': {},
        'result_dataframes': [df_res],
        'result_plots': []
    }


def test_key_metric_stability(model, sampler, scorer,
                              metric_name: str = 'gini',
                              abs_thresholds: tp.Tuple = (0.15, 0.25),
                              rel_thresholds: tp.Tuple = (0.20, 0.30),
                              greater_is_better: bool = True,
                              precomputed=None, **kwargs):
    if (precomputed is not None):
        if 'metrics_oos' in precomputed:
            metrics_oos = precomputed['metrics_oos']
        else:
            metrics_oos = get_source_metrics(
                model=model, sampler=sampler, scorer=scorer, data_type='oos')
            precomputed['metrics_oos'] = metrics_oos

        if 'metrics_oot' in precomputed:
            metrics_oot = precomputed['metrics_oot']
        else:
            metrics_oot = get_source_metrics(
                model=model, sampler=sampler, scorer=scorer, data_type='oot')
            precomputed['metrics_oot'] = metrics_oot

    res = report_test_key_metric_stability(
        metrics_oos, metrics_oot, metric_name, abs_thresholds, rel_thresholds, greater_is_better)
    return res
