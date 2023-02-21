import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from matplotlib import pyplot as plt

from sbe_vallib.utils.logger import Logger

TOP_COUNT = 20


def token_importance_test(
    sampler=None,
    X_sample=None,
    y_sample=None,
    vectorizer=None,
    path=".",
    test_name="test_token_importance",
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
    logger=None,
):
    save_excel = True if "tests_logs" in logging else False
    if logger is None:
        logger = Logger(path, level=logging)

    if X_sample is None:
        X_sample, y_sample, _ = sampler["train"]

    nan_values_mask = X_sample.isna().values
    X_sample = X_sample[~nan_values_mask]
    y_sample = y_sample[~nan_values_mask]

    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    X_sample_vectorized = vectorizer.fit_transform(X_sample)
    token_chi2_stats, _ = chi2(X_sample_vectorized, y_sample)
    tokens = np.array(vectorizer.get_feature_names())
    best_tokens_chi2 = tokens[token_chi2_stats.argsort()[::-1]][:TOP_COUNT]
    token_chi2_stats = np.sort(token_chi2_stats)[::-1][:TOP_COUNT]

    result = pd.DataFrame({'token': best_tokens_chi2, 'chi2': token_chi2_stats})

    plot_token_chi2(
        result,
        plt_filename='test_3_1_Token_importance',
        plt_show=False,
        path='.',
    )

    if save_excel:
        logger.save(result, test_name)

    test_res = "Grey"
    
    if "semaphore" in verbose:
        print("Результат теста {} : {}".format(test_name, test_res))

    if "short_res" in verbose:
        print("Тест {}".format(test_name))
        print(result.to_dict())

    logger("semaphore", "Результат теста {}: {}".format(test_name, test_res))
    logger("short_res", "Тест {}".format(test_name))
    logger("short_res", result.to_dict())
    
    return result


def plot_token_chi2(
        result_chi2, 
        plt_filename='test_3_1_Token_importance',
        plt_show=False, 
        path='.'
    ):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)

    ax.barh(result_chi2['token'], result_chi2['chi2'])

    ax.set_yticks(range(result_chi2.shape[0]))
    ax.set_yticklabels(result_chi2['token'].values)
    ax.invert_yaxis()
    ax.set_xlabel('$\chi ^ 2$ stat')
    ax.set_title(f'Top {result_chi2.shape[0]} token by $\chi ^ 2$ ')

    if plt_filename is not None:
        plt.savefig(os.path.join(path, plt_filename + '.png'))
    if plt_show:
        plt.show()

