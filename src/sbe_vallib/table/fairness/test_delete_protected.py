from copy import deepcopy

import pandas as pd

from sbe_vallib.utils.report_helper import worst_semaphore
from sbe_vallib.table.model_quality.test_key_metric import get_source_metrics


def report_delete_protected(main_metric, source_metrics, without_metrics, bad_features: list):
    col_source_metric = f'Исходная метрика {main_metric}'
    col_without_metric = f'Метрика {main_metric} после удаления'
    df = pd.DataFrame().from_records(index=[0], data={
        'Удаленные признаки': " ".join(bad_features),
        col_source_metric: source_metrics[main_metric],
        col_without_metric: without_metrics[main_metric]
    })
    df['Абс. изменение'] = df[col_without_metric] - df[col_source_metric]
    df['Отн. изменение'] = df['Абс. изменение'] / df[col_source_metric]
    return {
        'semaphore': 'gray',
        'result_dataframes': [df],
        'result_plots': [],
        'result_dict': None}


def select_bad_features(interest_tests, precomputed,
                        df_index=0, bad_semaphores=['yellow', 'red'],
                        feature_col='Защищенная характеристика',
                        semaphore_col='Результат теста'):
    all_df = []
    for test_key in interest_tests:
        df = precomputed[test_key]['result_dataframes'][df_index]
        all_df.append(df[[feature_col, semaphore_col]])
    df = pd.concat(all_df).groupby(by=feature_col)[
        semaphore_col].apply(worst_semaphore)
    return df[df.isin(bad_semaphores)].index.to_list()


def test_delete_protected(model, scorer, sampler, precomputed,
                          main_metric='gini', copy_model=True,
                          df_index=0, bad_semaphores=['yellow', 'red'],
                          feature_col='Защищенная характеристика',
                          semaphore_col='Результат теста', **kwargs):
    if precomputed is None:
        raise ValueError("""Test designed to excute only in pipeline and after all other fairness test,
                          precomputed should be not empty""")

    interest_tests = ['result_test_indirect_discrimination_gini',
                      'result_oppressed_privileged_ci',
                      'result_target_rate_delta',
                      'result_tprd_fprd_delta']
    for test_result in interest_tests:
        if test_result not in precomputed:
            raise ValueError(f"precomputed arg should contains {test_result}")

    bad_features = select_bad_features(
        interest_tests, precomputed, df_index, bad_semaphores, feature_col, semaphore_col)

    copy_sampler = deepcopy(sampler)
    if 'y_pred' in copy_sampler.train:
        del copy_sampler.train['y_pred']
    if 'y_pred' in copy_sampler.oos:
        del copy_sampler.oos['y_pred']
    copy_sampler.train['X'].drop(columns=bad_features, inplace=True)
    copy_sampler.oos['X'].drop(columns=bad_features, inplace=True)

    copy_model = deepcopy(model)
    copy_model.fit(copy_sampler.train['X'], copy_sampler.train['y_true'])

    source_metrics = get_source_metrics(
        model, sampler, scorer, data_type='oos')
    without_metrics = get_source_metrics(
        copy_model, copy_sampler, scorer, data_type='oos')

    result = report_delete_protected(
        main_metric, source_metrics, without_metrics, bad_features)
    return result
