import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer

from sbe_vallib.utility import load_light
from sbe_vallib.utils.logger import Logger

RANDOM_STATE = 42


def train_test_independence_test(
    sampler=None, 
    X_train=None,
    X_test=None,
    n_splits=4,
    n_iter=200,
    thresholds=(0.95, 0.99),
    path='.',
    recalc_indp=False,
    max_num_values=7_500_000,
    test_name='test_1_2_Train_test_independence',
    verbose=[
        "test_warnings",
        "test_errors",
        "semaphore",
        "model_errors",
        "model_warning",
        "progress",
        "short_res",
        "pictures",
        "info",
    ],
    logging=[
        "test_warnings",
        "test_errors",
        "semaphore",
        "model_errors",
        "model_warning",
        "progress",
        "short_res",
        "pictures",
        "answers",
        "tests_logs",
        "info",
    ],
    logger=None):

    save_excel = True if "tests_logs" in logging else False
    if logger is None:
        logger = Logger(path, level=logging)
    
    if (X_train is None) or (X_test is None):
        X_train, _, _ = sampler['train']
        X_test, _, _ = sampler['oos']

    if max_num_values is not None:
        total_values = (X_train.shape[0] + X_test.shape[0]) * X_train.shape[1]
        if total_values > max_num_values:
            frac = max_num_values / total_values
            print('\tTrain/oos samples are too huge for independence test, calculating on {:.4f} fraction'.format(frac))
            X_train = X_train.sample(frac=frac, random_state=RANDOM_STATE)
            X_test = X_test.sample(frac=frac, random_state=RANDOM_STATE)

    X_full = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_full = pd.Series(np.zeros((X_full.shape[0])))
    y_full[X_train.shape[0]:] = 1

    gini_fact, gini_random_list = calculate_train_test_independence(X_full, y_full, n_splits, n_iter, 
                                                                            recalc_indp, logger, verbose)

    gini_random_list = sorted(gini_random_list)
    gini_quantile = min(sum(gini_random_list <= gini_fact) / (len(gini_random_list) - 1), 1)
    result_color = get_independence_result_color(gini_quantile, thresholds)

    plot_results(
        gini_fact,
        gini_random_list,
        gini_quantile,
        result_color,
        plt_filename='test_1_2_Train_test_independence',
        plt_show=False,
        path='.',
    )

    result = pd.DataFrame({'Фактический gini': gini_fact,
                           'Квантиль': gini_quantile,
                           'Результат теста': result_color
                           }, index=[0])
    
    if save_excel:
        logger.save(result, test_name)
    
    test_res = result['Результат теста'][0]
    
    if "semaphore" in verbose:
        print("Результат теста {} : {}".format(test_name, test_res))

    if "short_res" in verbose:
        print("Тест {}".format(test_name))
        print(result.to_dict())

    logger("semaphore", "Результат теста {}: {}".format(test_name, test_res))
    logger("short_res", "Тест {}".format(test_name))
    logger("short_res", result.to_dict())
    
    return result


def calculate_train_test_independence(X_full, y_full, n_splits, n_iter, recalc_indp, logger, verbose):
    model = LGBMClassifier(n_jobs=1)
    gini_scorer = make_scorer(lambda x, y: -1 + 2 * roc_auc_score(x, y), needs_proba=True)

    add_path = 'split_independence'
    logger.checkOrCreate(add_path)
    indep_true = 'indep_true'
    gini_fact = logger.load(indep_true, add_path=add_path)

    if (gini_fact is None) or recalc_indp:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        gini_fact = cross_val_score(model, X_full, y_full, cv=kfold, scoring=gini_scorer, n_jobs=1).mean()
        logger.dump(gini_fact, name_artifact=indep_true, add_path=add_path)

    if 'progress' in verbose:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x
    
    gini_random_list = []
    for i in tqdm(range(n_iter)):
        gini_iter = logger.load(f'indep_random_{i}', add_path=add_path)
        if (gini_iter is None) or recalc_indp:
            y_full_shuffled = y_full.sample(frac=1, replace=False, random_state=i)
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
            gini_iter = cross_val_score(model, X_full, y_full_shuffled, cv=kfold, scoring=gini_scorer, n_jobs=1).mean()
            logger.dump(gini_iter, name_artifact=f'indep_random_{i}', add_path=add_path)
        gini_random_list.append(gini_iter)
    
    return gini_fact, gini_random_list


def get_independence_result_color(value, thresholds):
    quantile_yellow, quantile_red = thresholds
    if (1 - quantile_yellow) / 2 < value < (1 + quantile_yellow) / 2:
        result_color = "Green"
    elif (1 - quantile_red) / 2 <= value <= (1 + quantile_red) / 2:
        result_color = "Yellow"
    else:
        result_color = "Red"
    return result_color


def plot_results(
        gini_fact,
        gini_random_list,
        gini_quantile,
        result_color,
        plt_filename='test_1_2_Train_test_independence',
        plt_show=False,
        path='.'):

    x_lim_min = min(gini_fact, min(gini_random_list)) * 1.05
    x_lim_max = max(gini_fact, max(gini_random_list)) * 1.05

    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches(20, 10)

    density_plot = plt.subplot2grid((20, 20), (0, 0), colspan=16, rowspan=10)
    density_plot.set_xticks([])
    density_plot.set_xlim(x_lim_min, x_lim_max)

    density_plot.plot([gini_fact, gini_fact], [0, 0.25], "--", color=result_color, linewidth=2)

    weights = np.ones(len(gini_random_list)) / len(gini_random_list)
    density_plot_params = density_plot.hist(gini_random_list, weights=weights, bins=100, alpha=0.5,
                                            label='Gini density for random splits', color='Orange')

    y_lim_max = max(density_plot_params[0]) * 1.25
    density_plot.set_ylim(0, y_lim_max)
    density_plot.set_ylabel('Probability')

    distrib_plot = plt.subplot2grid((20, 20), (10, 0), colspan=16, rowspan=10)
    distrib_plot.set_xlim(x_lim_min, x_lim_max)

    distrib_plot.plot(gini_random_list, np.linspace(0, 1, len(gini_random_list)),
                      label='Gini distribution for random splits')

    distrib_plot.plot([gini_fact] * 2, [0, 1], "--", color=result_color,
                      label="Fact gini(%2.2f), quantile(%2.2f)" % (gini_fact, gini_quantile))

    distrib_plot.plot([min(gini_random_list), max(max(gini_random_list), gini_fact)],
                      [gini_quantile, gini_quantile], "--", color=result_color)

    distrib_plot.scatter(gini_fact, gini_quantile, marker='x', s=100, color='red')
    distrib_plot.set_xlabel('Gini')
    distrib_plot.set_ylabel('Probability')
    distrib_plot.grid()

    light_plot = plt.subplot2grid((20, 20), (0, 16), colspan=4, rowspan=4)
    light_plot.axis('off')

    text_plot = plt.subplot2grid((20, 20), (4, 17), colspan=16, rowspan=4)
    text_plot.text(0, 0, "Fact gini = %2.3f \nQuantile  = %2.3f" % (gini_fact, gini_quantile))
    text_plot.axis('off')

    light_plot.imshow(load_light(result_color))

    fig.legend(plt_filename, plt_filename, loc=(0.8, 0.2))
    if plt_filename is not None:
        plt.savefig(os.path.join(path, plt_filename), transparent=False)
    if plt_show:
        plt.show()

    plt.close()