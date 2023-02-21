import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

# from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sbe_vallib.validation.utils import concat, get_index

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

gini_scorer = make_scorer(lambda x, y: -1 + 2 * roc_auc_score(x, y), needs_proba=True)

RANDOM_STATE = 42


def train_test_independence_test(
    sampler,
    n_splits=4,
    n_iter=20,
    thresholds=(0.95, 0.99),
    recalc_indp=False,
    max_num_values=7_500,
    **kwargs,
):

    sampler.reset()

    size_of_train = len(sampler.train["X"])
    size_of_oos = len(sampler.oos["X"])

    if max_num_values is not None:
        total_values = size_of_train + size_of_oos
        if total_values > max_num_values:

            generator = np.random.default_rng(seed=42)
            generator.integers(0, size_of_oos, size_of_oos)

            frac = max_num_values / total_values

            print(
                "\tTrain/oos samples are too huge for independence test, calculating on {:.4f} fraction".format(
                    frac
                )
            )
            X_full = concat(
                [
                    get_index(
                        sampler.train["X"],
                        generator.integers(0, size_of_train, int(size_of_train * frac)),
                    ),
                    get_index(
                        sampler.oos["X"],
                        generator.integers(0, size_of_oos, int(size_of_oos * frac)),
                    ),
                ]
            )
            y_full = pd.Series(np.zeros(len(X_full)))
            y_full[int(size_of_train * frac):] = 1

    else:
        X_full = concat([sampler.train["X"], sampler.oos["X"]])
        y_full = pd.Series(np.zeros(len(sampler.train["X"]) + len(sampler.oos["X"])))
        y_full[len(sampler.train["X"]) :] = 1
        
    gini_fact, gini_random_list = calculate_train_test_independence(
        X_full, y_full, n_splits, n_iter
    )

    gini_random_list = sorted(gini_random_list)
    gini_quantile = min(
        sum(gini_random_list <= gini_fact) / (len(gini_random_list) - 1), 1
    )
    result_color = "Grey"  # get_independence_result_color(gini_quantile, thresholds)

    result = {
        "Фактический gini": gini_fact,
        "Квантиль": gini_quantile,
        "Результат теста": result_color,
    }

    return {
        "semaphore": "grey",
        "result_dict": {"result": result},
        "result_dataframes": [pd.DataFrame(result, index=[0])],
        "result_plots": [],
    }


def calculate_train_test_independence(X_full, y_full, n_splits, n_iter):

    model = LogisticRegression()

    vectorizer = TfidfVectorizer(min_df=50, max_df=80)
    X_full = vectorizer.fit_transform(X_full)

    print(X_full.shape)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    gini_fact = cross_val_score(
        model, X_full, y_full, cv=kfold, scoring=gini_scorer
    ).mean()

    gini_random_list = []
    for i in tqdm(range(n_iter)):

        y_full_shuffled = y_full.sample(frac=1, replace=False, random_state=i)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)

        gini_iter = cross_val_score(
            model, X_full, y_full_shuffled, cv=kfold, scoring=gini_scorer
        ).mean()

        gini_random_list.append(gini_iter)

    return gini_fact, gini_random_list


# def get_independence_result_color(value, thresholds):
#     quantile_yellow, quantile_red = thresholds
#     if (1 - quantile_yellow) / 2 < value < (1 + quantile_yellow) / 2:
#         result_color = "Green"
#     elif (1 - quantile_red) / 2 <= value <= (1 + quantile_red) / 2:
#         result_color = "Yellow"
#     else:
#         result_color = "Red"
#     return result_color


# def plot_results(
#         gini_fact,
#         gini_random_list,
#         gini_quantile,
#         result_color,
#         plt_filename='test_1_2_Train_test_independence',
#         plt_show=False,
#         path='.'):

#     x_lim_min = min(gini_fact, min(gini_random_list)) * 1.05
#     x_lim_max = max(gini_fact, max(gini_random_list)) * 1.05

#     plt.close('all')
#     fig = plt.figure()
#     fig.set_size_inches(20, 10)

#     density_plot = plt.subplot2grid((20, 20), (0, 0), colspan=16, rowspan=10)
#     density_plot.set_xticks([])
#     density_plot.set_xlim(x_lim_min, x_lim_max)

#     density_plot.plot([gini_fact, gini_fact], [0, 0.25], "--", color=result_color, linewidth=2)

#     weights = np.ones(len(gini_random_list)) / len(gini_random_list)
#     density_plot_params = density_plot.hist(gini_random_list, weights=weights, bins=100, alpha=0.5,
#                                             label='Gini density for random splits', color='Orange')

#     y_lim_max = max(density_plot_params[0]) * 1.25
#     density_plot.set_ylim(0, y_lim_max)
#     density_plot.set_ylabel('Probability')

#     distrib_plot = plt.subplot2grid((20, 20), (10, 0), colspan=16, rowspan=10)
#     distrib_plot.set_xlim(x_lim_min, x_lim_max)

#     distrib_plot.plot(gini_random_list, np.linspace(0, 1, len(gini_random_list)),
#                       label='Gini distribution for random splits')

#     distrib_plot.plot([gini_fact] * 2, [0, 1], "--", color=result_color,
#                       label="Fact gini(%2.2f), quantile(%2.2f)" % (gini_fact, gini_quantile))

#     distrib_plot.plot([min(gini_random_list), max(max(gini_random_list), gini_fact)],
#                       [gini_quantile, gini_quantile], "--", color=result_color)

#     distrib_plot.scatter(gini_fact, gini_quantile, marker='x', s=100, color='red')
#     distrib_plot.set_xlabel('Gini')
#     distrib_plot.set_ylabel('Probability')
#     distrib_plot.grid()

#     light_plot = plt.subplot2grid((20, 20), (0, 16), colspan=4, rowspan=4)
#     light_plot.axis('off')

#     text_plot = plt.subplot2grid((20, 20), (4, 17), colspan=16, rowspan=4)
#     text_plot.text(0, 0, "Fact gini = %2.3f \nQuantile  = %2.3f" % (gini_fact, gini_quantile))
#     text_plot.axis('off')

#     light_plot.imshow(load_light(result_color))

#     fig.legend(plt_filename, plt_filename, loc=(0.8, 0.2))
#     if plt_filename is not None:
#         plt.savefig(os.path.join(path, plt_filename), transparent=False)
#     if plt_show:
#         plt.show()

#     plt.close()
