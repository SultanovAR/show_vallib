import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sbe_vallib.utils.image import plt2PIL


def create_distr_image(metrics_train, metrics_test, plt_title,
                       quantile=0.95, plt_quantile_height=0.01, plt_xlim=None, plt_ylim=(0, 0.1),
                       plt_figsize=(15, 5), plt_bins=100):
    # Format plot
    fig = plt.figure()
    fig.set_size_inches(*plt_figsize)

    plt.title(plt_title, fontsize=14)
    plt.ylabel("% of evidence")
    plt.ylim(*plt_ylim)

    if plt_xlim is None:
        min_ = min(min(metrics_train), min(metrics_test))
        min_ = min_ * (0.9 if min_ > 0 else 1.1)

        max_ = max(max(metrics_train), max(metrics_test))
        if 0 <= max_ <= 1:
            max_ = min(max_ * 1.1, 1.001)
        else:
            max_ = max_ * 1.1

        plt.xlim(min_, max_)
    else:
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
    left_q = 1 / 2 - quantile / 2
    right_q = 1 / 2 + quantile / 2

    left_quantile_train = sorted(metrics_train)[
        int(left_q * len(metrics_train))]
    right_quantile_train = sorted(metrics_train)[
        int(right_q * len(metrics_train))]
    plt.plot(
        [
            left_quantile_train,
            left_quantile_train,
            right_quantile_train,
            right_quantile_train,
        ],
        [0, plt_quantile_height, plt_quantile_height, 0],
        label="Train %0.2f confidence interval" % quantile,
        color="red",
        linestyle="dashed",
    )

    left_quantile_test = sorted(metrics_test)[int(left_q * len(metrics_test))]
    right_quantile_test = sorted(metrics_test)[
        int(right_q * len(metrics_test))]
    plt.plot(
        [
            left_quantile_test,
            left_quantile_test,
            right_quantile_test,
            right_quantile_test,
        ],
        [0, plt_quantile_height, plt_quantile_height, 0],
        label="Test %0.2f confidence interval" % quantile,
        color="blue",
        linestyle="dashed",
    )
    pil_image = plt2PIL(fig)
    plt.close()
    return pil_image


def _get_ci(data, quantile):
    left_q = 1 / 2 - quantile / 2
    right_q = 1 / 2 + quantile / 2
    return data[int(left_q * len(data))], data[int(right_q * len(data))]


def report_test_ci(metrics_stat: dict, metric_name: str, quantile: float):
    res = {'semaphore': 'gray', 'result_dataframes': [], 'result_plots': []}
    metrics_train = np.array(
        sorted([i[metric_name] for i in metrics_stat['train']]))
    metrics_test = np.array(
        sorted([i[metric_name] for i in metrics_stat['oos']]))

    metric_diff_abs = sorted(100 * (metrics_train - metrics_test))
    metric_diff_rel = sorted(
        100 * (metrics_train - metrics_test) / metrics_train)
    left_q_train, right_q_train = _get_ci(metrics_train, quantile)
    left_q_test, right_q_test = _get_ci(metrics_test, quantile)
    left_q_abs, right_q_abs = _get_ci(metric_diff_abs, quantile)
    left_q_rel, right_q_rel = _get_ci(metric_diff_rel, quantile)

    res['result_plots'].append(create_distr_image(
        metrics_train, metrics_test, metric_name))
    res['result_dataframes'].append(
        pd.DataFrame(index=["test", "train", "Absolute difference (p.p.)", "Relative difference (%%)"],
                     data={
            "mean": [np.mean(metrics_test), np.mean(metrics_train),
                     np.mean(metric_diff_abs), np.mean(metric_diff_rel)],
            "std": [np.std(metrics_test), np.std(metrics_train),
                    np.std(metric_diff_abs), np.std(metric_diff_rel)],
            "left bound": [left_q_test, left_q_train,
                           left_q_abs, left_q_rel],
            "right bound": [right_q_test, right_q_train,
                            right_q_abs, right_q_rel]}
        )
    )
    res['result_dataframes'].append(
        pd.DataFrame([{f'Значение метрики {metric_name}': metrics_test.mean(),
                       'Довер-ный интервал OOS': [left_q_test, right_q_test],
                       'Количество наблюдений OOS': len(metrics_test)}])
    )
    return res


def test_ci(model, sampler, scorer,
            metric_name='gini', n_iter=200,
            gen_method='resampling', quantile=0.95, **kwargs):
    sampler.reset()
    if gen_method == 'bootstrap':  # when we apply a bootstrap version a model doesn't refit
        if 'y_pred' not in sampler.oos:
            sampler.oos['y_pred'] = model.predict(sampler.oos['X'])
        if 'y_pred' not in sampler.train:
            sampler.train['y_pred'] = model.predict(sampler.train['X'])

    metrics_stat = {'train': [], 'oos': []}
    for i in range(n_iter):
        sampler.set_state(seed=i, gen_method=gen_method)
        if sampler._gen_method == 'bootstrap':
            metrics_stat['train'].append(
                scorer.calc_metrics(y_true=sampler.train['y_true'], y_proba=sampler.train['y_pred']))
            metrics_stat['oos'].append(
                scorer.calc_metrics(y_true=sampler.oos['y_true'], y_proba=sampler.oos['y_pred']))
        else:
            model.fit(X=sampler.train["X"], y=sampler.train["y_true"])
            metrics_stat['train'].append(
                scorer.calc_metrics(model, scorer, data_type='train'))
            metrics_stat['oos'].append(
                scorer.calc_metrics(model, scorer, data_type='oos'))

    result = {'result_dict': {}}
    result.update(report_test_ci(metrics_stat, metric_name, quantile))
    return result
