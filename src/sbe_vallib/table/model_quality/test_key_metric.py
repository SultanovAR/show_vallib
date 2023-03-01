import typing as tp

import pandas as pd

from sbe_vallib.utils.report_sberds import semaphore_by_threshold


def get_source_metrics(model, sampler, scorer, data_type='oos'):
    sampler.reset()
    data = getattr(sampler, data_type)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])
    metrics = scorer.calc_metrics(
        model=model, sampler=sampler, data_type=data_type, use_preds_from_sampler=True)
    return metrics


def report_test_key_metric(metrics, metric_name: str, thresholds: tp.Tuple, greater_is_better: bool):
    semaphore = semaphore_by_threshold(
        metrics[metric_name], thresholds, greater_is_better)
    df_res = pd.DataFrame(
        [{metric_name: metrics[metric_name], 'semaphore': semaphore}])
    return {
        'semaphore': semaphore,
        'result_dataframes': [df_res],
        'result_plots': []
    }


def test_key_metric(model, sampler, scorer,
                    metric_name: str = 'gini',
                    thresholds: tp.Tuple = (0.4, 0.6),
                    greater_is_better: bool = True,
                    data_type: str = 'oos',
                    precomputed=None, **kwargs):
    if (precomputed is not None):
        if f'metrics_{data_type}' in precomputed:
            metrics = precomputed[f'metrics_{data_type}']
        else:
            metrics = get_source_metrics(
                model=model, sampler=sampler, scorer=scorer, data_type=data_type)
            precomputed[f'metrics_{data_type}'] = metrics

    res_dict = report_test_key_metric(
        metrics, metric_name, thresholds, greater_is_better)
    res_dict.update({'result_dict': {'metrics': metrics}})
    return res_dict
