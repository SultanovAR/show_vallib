import typing as tp

import pandas as pd
import numpy as np

from sbe_vallib.utils.report_helper import semaphore_by_threshold


def aggregate_semaphore_key_metric(metric_avg: float, semaphores: tp.List[str],
                                   thresholds: tp.Tuple):
    uniq, counts = np.unique(semaphores, return_counts=True)
    freq = {semaphore: count / len(semaphores)
            for semaphore, count in zip(uniq, counts)}

    if (metric_avg < thresholds[0])\
            or (freq.get('red', 0) > 0.2):
        return 'red'
    elif (thresholds[0] < metric_avg < thresholds[1])\
            or (freq.get('yellow', 0) > 0.2)\
            or (freq.get('red', 0) > 0):
        return 'yellow'
    else:
        return 'green'


def report_key_metric(metrics: tp.Dict, metric_name: str, classes: tp.List[str],
                      average: str, min_support=20, thresholds=(0.4, 0.6)):
    table = []
    if (classes is not None) and (isinstance(metrics[metric_name], dict)):
        for key in classes:
            if metrics['support'][key] < min_support:
                semaphore = 'gray'
            else:
                semaphore = semaphore_by_threshold(
                    metrics[metric_name][key], thresholds)
            table.append({'type': key,
                          'support': metrics['support'][key],
                          metric_name: metrics[metric_name][key],
                          'semaphore': semaphore})

        semaphores = [i['semaphore'] for i in table]
        agregated_semaphore = aggregate_semaphore_key_metric(
            metrics[metric_name][average], semaphores, thresholds)
        table.append({'type': f'agg_{average}',
                      'support': metrics['support'][average],
                      metric_name: metrics[metric_name][average],
                      'semaphore': agregated_semaphore})
    else:
        agregated_semaphore = semaphore_by_threshold(
            metrics[metric_name], thresholds)
        table.append({'type': f'agg_{average}',
                      'support': metrics['support'],
                      metric_name: metrics[metric_name],
                      'semaphore': agregated_semaphore})

    df_table = pd.DataFrame.from_records(table)
    return agregated_semaphore, df_table


def test_key_metric(model, scorer, sampler,
                    metric_name='precision_score', average='macro', data_type='oos',
                    thresholds=(0.4, 0.6), min_support=20, use_preds_from_sampler=True, **kwargs):
    sampler.reset()
    data = getattr(sampler, data_type)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])

    metrics = scorer.calc_metrics(model=model,
                                  sampler=sampler,
                                  data_type=data_type,
                                  use_preds_from_sampler=use_preds_from_sampler, ** kwargs)

    semaphore, df_report = report_key_metric(
        metrics, metric_name, getattr(model, 'classes_', None), average, min_support, thresholds)
    return {
        "semaphore": semaphore,
        "result_dict": {"metrics": metrics},
        "result_dataframes": [df_report],
        "result_plots": [],
    }
