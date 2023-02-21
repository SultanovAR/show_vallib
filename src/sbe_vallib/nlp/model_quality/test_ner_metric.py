import typing as tp

import pandas as pd
import numpy as np

from sbe_vallib.utils.report_sberds import semaphore_by_threshold


def report_ner_metric(metrics: tp.Dict):
    df_table = pd.DataFrame(metrics).reset_index(
        names='Наименование класса', inplace=False)
    return 'gray', df_table


def test_ner_metric(model, scorer, sampler, type_data='oos', **kwargs):
    sampler.reset()
    data = getattr(sampler, type_data)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])

    metrics = scorer.ner_metrics(y_true=data['y_true'],
                                 y_pred=data['y_pred'],
                                 model=model,
                                 sampler=sampler, **kwargs)

    semaphore, df_report = report_ner_metric(metrics)
    return {
        "semaphore": semaphore,
        "result_dict": {"metrics": metrics},
        "result_dataframes": [df_report],
        "result_plots": [],
    }
