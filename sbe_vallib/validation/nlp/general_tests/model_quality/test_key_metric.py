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


def report_key_metric(metrics: tp.Dict, classes: tp.List[str], average: str, 
                      metric_name: str, min_support: int, thresholds: tp.Tuple):
    table = []
    for  in classes:

        if metrics['support'][tag] < min_support:
            semaphore = 'gray'
        else:
            semaphore = semaphore_by_threshold(
                metrics[f'{metric_name}_by_{tag}'], thresholds)
        table.append({'tag': tag,
                      'support': metrics[f'support_by_{tag}'],
                      f'{metric_name}': metrics[f'{metric_name}_by_{tag}'],
                      'semaphore': semaphore})

    metric_avg = metrics[metric_name][average]
    semaphores = [i['semaphore'] for i in table]
    agregated_semaphore = aggregate_semaphore_key_metric(
        metric_avg, semaphores, thresholds)
    table.append({
        'tag': average,
        'support': metrics['support'][average],
        f'{metric_name}': metric_avg,
        'semaphore': agregated_semaphore
    })

    df_table = pd.DataFrame.from_records(table)
    return agregated_semaphore, df_table


def test_key_metric(model, scorer, sampler, type_data='oos', average='macro',
                    metric_name='f1-score', thresholds=(0.4, 0.6), min_support=20, **kwargs):
    sampler.reset()
    data = getattr(sampler, type_data)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])

    metrics = scorer.score(y_true=data['y_true'],
                           y_pred=data['y_pred'],
                           model=model,
                           sampler=sampler, **kwargs)

    semaphore, df_report = report_key_metric(
        metrics, scorer._classes, average, metric_name, min_support, thresholds)
    return {
        "semaphore": semaphore,
        "result_dict": {"metrics": metrics},
        "result_dataframes": [df_report],
        "result_plots": [],
    }
