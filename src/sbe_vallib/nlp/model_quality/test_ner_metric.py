import typing as tp

import pandas as pd



def report_ner_metric(metrics: tp.Dict):
    df_table = pd.DataFrame(metrics).reset_index(
        names='Наименование класса', inplace=False)
    return 'gray', df_table


def test_ner_metric(model, scorer, sampler, data_type='oos', **kwargs):
    sampler.reset()
    data = getattr(sampler, data_type)
    if 'y_pred' not in data:
        data['y_pred'] = model.predict(data['X'])

    metrics = scorer.ner_metrics(model=model,
                                 sampler=sampler,
                                 data_type=data_type,
                                 use_preds_from_sampler=True, **kwargs)

    semaphore, df_report = report_ner_metric(metrics)
    return {
        "semaphore": semaphore,
        "result_dict": {"metrics": metrics},
        "result_dataframes": [df_report],
        "result_plots": [],
    }
