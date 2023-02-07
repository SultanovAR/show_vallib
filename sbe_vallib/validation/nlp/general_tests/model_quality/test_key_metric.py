import typing as tp

import pandas as pd
import numpy as np

from sbe_vallib.validation.utils.report_sberds import semaphore_by_threshold


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
    if isinstance(metrics[metric_name], dict):  # multiclass
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
    else:  # float
        agregated_semaphore = semaphore_by_threshold(
            metrics[metric_name], thresholds)
        table.append({'type': f'agg_{average}',
                      'support': metrics['support'],
                      metric_name: metrics[metric_name],
                      'semaphore': agregated_semaphore})

    df_table = pd.DataFrame.from_records(table)
    return agregated_semaphore, df_table


def test_key_metric(model, scorer, sampler, metric_name,
                    classes, average, type_data='oos',
                    thresholds=(0.4, 0.6), min_support=20, **kwargs):
    sampler.reset()
    data = getattr(sampler, type_data)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])

    metrics = scorer.score(y_true=data['y_true'],
                           y_pred=data['y_pred'],
                           model=model,
                           sampler=sampler, **kwargs)

    semaphore, df_report = report_key_metric(
        metrics, metric_name, classes, average, min_support, thresholds)
    return {
        "semaphore": semaphore,
        "result_dict": {"metrics": metrics},
        "result_dataframes": [df_report],
        "result_plots": [],
    }
