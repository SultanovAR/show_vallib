from collections import defaultdict
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sbe_vallib.utils.image import plt2PIL


def _get_ci(data, quantile):
    left_q = 1 / 2 - quantile / 2
    right_q = 1 / 2 + quantile / 2
    return np.quantile(data, q=left_q), np.quantile(data, q=right_q)


def _get_plt_xlim(metrics_train, metrics_test):
    left_lim = min(min(metrics_train), min(metrics_test))
    left_lim = left_lim * (0.9 if left_lim > 0 else 1.1)

    right_lim = max(max(metrics_train), max(metrics_test))
    if 0 <= right_lim <= 1:
        right_lim = min(right_lim * 1.1, 1.001)
    else:
        right_lim = right_lim * 1.1
    return left_lim, right_lim


def create_distr_image(metrics_train, metrics_test, plt_title, quantile,
                       plt_quantile_height=0.01, plt_xlim=None, plt_ylim=(0, 0.1),
                       plt_figsize=(15, 5), plt_bins=100):
    # Format plot
    fig = plt.figure(figsize=plt_figsize)
    plt.title(plt_title, fontsize=14)
    plt.ylabel("% of evidence")
    plt.ylim(*plt_ylim)
    if plt_xlim is None:
        plt_xlim = _get_plt_xlim(metrics_train, metrics_test)
    plt.xlim(*plt_xlim)

    # Draw histograms for train and test
    bins = np.linspace(
        min(min(metrics_test), min(metrics_train)),
        max(max(metrics_test), max(metrics_train)),
        plt_bins,
    )

    plt.hist(
        metrics_train,
        bins=bins,
        label="Train",
        alpha=0.7,
        weights=np.ones_like(metrics_train) / float(len(metrics_train)),
    )

    plt.hist(
        metrics_test,
        bins=bins,
        label="Test",
        alpha=0.7,
        weights=np.ones_like(metrics_train) / float(len(metrics_train)),
    )

    # Draw quantiles
    quantiles = {}
    quantiles['train'] = _get_ci(metrics_train, quantile)
    plt.plot([quantiles['train'][i] for i in [0, 0, 1, 1]],
             [0, plt_quantile_height, plt_quantile_height, 0],
             label="Train %0.2f confidence interval" % quantile,
             color="red",
             linestyle="dashed")

    quantiles['test'] = _get_ci(metrics_test, quantile)
    plt.plot([quantiles['test'][i] for i in [0, 0, 1, 1]],
             [0, plt_quantile_height, plt_quantile_height, 0],
             label="Test %0.2f confidence interval" % quantile,
             color="blue",
             linestyle="dashed")
    plt.legend()
    pil_image = plt2PIL(fig)
    plt.close()
    return pil_image


def report_test_ci(metrics: dict, metric_name: str, q_lvl: float):
    res = {'semaphore': 'gray', 'result_dataframes': [], 'result_plots': []}
    metrics = {key: np.array(metrics[key]) for key in metrics}
    metrics['abs'] = np.array(100 * (metrics['train'] - metrics['oos']))
    metrics['rel'] = np.array(
        metrics['train'] - metrics['oos'] / metrics['train'])

    quantiles = {}
    quantiles['train'] = _get_ci(metrics['train'], q_lvl)
    quantiles['oos'] = _get_ci(metrics['oos'], q_lvl)
    quantiles['abs'] = _get_ci(metrics['abs'], q_lvl)
    quantiles['rel'] = _get_ci(metrics['rel'], q_lvl)

    res['result_dataframes'].append(
        pd.DataFrame(index=["test", "train", "Absolute difference (p.p.)", "Relative difference (%%)"],
                     data={
            "mean": [metrics[key].mean() for key in ['oos', 'train', 'abs', 'rel']],
            "std": [metrics[key].std() for key in ['oos', 'train', 'abs', 'rel']],
            "left bound": [quantiles[key][0] for key in ['oos', 'train', 'abs', 'rel']],
            "right bound": [quantiles[key][1] for key in ['oos', 'train', 'abs', 'rel']]}
        )
    )
    res['result_dataframes'].append(
        pd.DataFrame([{f'Значение метрики {metric_name}': metrics['oos'].mean(),
                       'Довер-ный интервал OOS': quantiles['oos'],
                       'Количество наблюдений OOS': len(metrics['oos'])}])
    )

    res['result_plots'].append(create_distr_image(
        metrics['train'], metrics['oos'], metric_name, q_lvl))
    return res


def test_ci(model, sampler, scorer,
            metric_name='gini', n_iter=200,
            gen_method='resampling', quantile=0.95, **kwargs):
    if gen_method == 'bootstrap':  # when we apply a bootstrap version a model doesn't refit
        sampler.reset()
        if 'y_pred' not in sampler.oos:
            sampler.oos['y_pred'] = model.predict(sampler.oos['X'])
        if 'y_pred' not in sampler.train:
            sampler.train['y_pred'] = model.predict(sampler.train['X'])
        model_ci = model
        use_preds_from_sampler = True
    else:
        use_preds_from_sampler = False
        model_ci = deepcopy(model)

    metric_stat = defaultdict(list)
    for i in range(n_iter):
        sampler.set_state(seed=i, gen_method=gen_method, stratify=True)
        if sampler._gen_method != 'bootstrap':
            model_ci.fit(X=sampler.train["X"], y=sampler.train["y_true"])
        train_metrics = scorer.calc_metrics(model=model_ci,
                                            sampler=sampler,
                                            data_type='train',
                                            use_preds_from_sampler=use_preds_from_sampler)
        oos_metrics = scorer.calc_metrics(model=model_ci,
                                          sampler=sampler,
                                          data_type='oos',
                                          use_preds_from_sampler=use_preds_from_sampler)
        metric_stat['train'].append(train_metrics[metric_name])
        metric_stat['oos'].append(oos_metrics[metric_name])

    result = report_test_ci(metric_stat, metric_name, quantile)
    result.update({'result_dict': {}})
    return result
