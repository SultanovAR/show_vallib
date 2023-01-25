import numpy as np
import pandas as pd
import math
import typing as tp

from scipy.special import rel_entr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sbe_vallib.validation.sampler import BinarySampler
from sbe_vallib.validation.utils import pd_np_interface as interface
from sbe_vallib.validation.utils.quantization import Quantization
from sbe_vallib.validation.utils.report_sberds import semaphore_by_threshold, worst_semaphore


def psi(train, oos, base=None, axis=0):
    EPS = 1e-7
    train = train / np.sum(train) + EPS
    oos = oos / np.sum(oos) + EPS
    return np.sum(rel_entr(train, oos) + rel_entr(oos, train))


def report_factor_psi(psi_of_feat: tp.Dict, threshold: tp.Tuple):
    """Create dict with the SberDS format

    Parameters
    ----------
    psi_of_feat : tp.Dict
        dictionary with the following format {'feature': psi_value}
    """

    semaphores = {feat: semaphore_by_threshold(
        psi_of_feat[feat]['psi'], threshold, False) for feat in psi_of_feat}

    res_df = psi_of_feat
    for feat in psi_of_feat:
        res_df[feat]['feature'] = feat
        res_df[feat]['hist_train'] = [
            f'{i:.3f}' for i in psi_of_feat[feat]['hist_train']]
        res_df[feat]['hist_oos'] = [
            f'{i:.3f}' for i in psi_of_feat[feat]['hist_oos']]
        res_df[feat]['semaphore'] = semaphores[feat]

    result = dict()
    result['semaphore'] = worst_semaphore(semaphores.values())
    result['result_dict'] = psi_of_feat
    result['result_dataframes'] = pd.DataFrame.from_records(res_df)
    result['result_plots'] = []
    return result


def get_feature_types(data, discr_uniq_val=10, discr_val_share=0.8):
    """
    Determine type of every DataFrame column: discrete or continuous.
    If the number of observations with discr_uniq_val most common unique values
        is more than discr_val_share of total sample, column is considered discrete.

    :param df: Input DataFrame
    :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
    :param discr_val_share: share of most common values that is enough to define factor as discrete
    :return: dict of {'feature': 'Discrete' or 'Continuous'}
    """
    data_dropna = np.array(data)[~np.isnan(data)]
    # Get sorted values by counts descending
    unique_counts = np.unique(data_dropna, return_counts=True)[1]
    unique_counts = sorted(unique_counts, reverse=True)

    if np.sum(unique_counts[:discr_uniq_val]) / len(data_dropna) >= discr_val_share:
        return 'Discrete'
    return 'Continuous'


def test_factor_psi(sampler,
                    merge_upto_quantile: float = 0.05,
                    rounding_precision_bins: int = 5,
                    discr_uniq_val: int = 10,
                    discr_val_share: float = 0.8,
                    threshold: tp.Tuple = (0.1, 0.3), **kwargs):
    sampler.reset()
    x_train, x_oos = sampler.train['X'], sampler.oos['X']

    encoder = LabelEncoder()
    quantizer = Quantization(merge_upto_quantile, rounding_precision_bins)

    psi_of_feat = {}
    for col in interface.all_columns(x_oos):
        feat_type = get_feature_types(interface.get_columns(x_train, col),
                                      discr_uniq_val, discr_val_share)
        train_col = np.array(interface.get_columns(x_train, col))[:, None]
        oos_col = np.array(interface.get_columns(x_oos, col))[:, None]
        if feat_type == 'Discrete':
            train_col = encoder.fit_transform(train_col)
            oos_col = encoder.transform(oos_col)
        bins = quantizer.fit(train_col).bins[0]

        hist_train_col, _ = np.histogram(train_col, bins=bins)
        hist_oos_col, _ = np.histogram(oos_col, bins=bins)
        psi_of_feat[col] = {
            'psi': psi(hist_train_col, hist_oos_col),
            'feat_type': feat_type,
            'bin_count': len(bins),
            'hist_train': hist_train_col,
            'hist_oos': hist_oos_col,
            'bins': bins
        }

    return report_factor_psi(psi_of_feat, threshold)


# def test_1_4(discr_uniq_val=10,
#              discr_val_share=0.8,
#              min_bin_size=0.05,
#              path='.',
#              save_excel=False,
#              debug=False,
#              **kwargs):
#     """
#     Calculate PSI for each factor on X_train and X_test. Uses dump data.
#     Executes function psi_factor_test().

#     Two types of factors are defined: continious and discrete.
#     To find out which type the factor is we take discr_uniq_val most common unique values.
#     If number of observations with these values is more than discr_val_share of total sample, factor is labeled discrete.
#     Otherwise it is considered continuous.

#     Type of factor defines binning algorithm used to calculate PSI.

#     :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
#     :param discr_val_share: share of most common values that is enough to define factor as discrete
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :param path: relative or absolute path for output file
#     :param save_excel: whether or not to save excel file (overwrite if already exists)
#     :param debug: turn on debug mode, adding technical info to resulting DataFrame
#     :return: DataFrame with PSI stats by factor
#     """
#     X_train = load_dump('X_train_proc', path)
#     X_test = load_dump('X_test_proc', path)

#     result_oos = psi_factor_test(X_train, X_test, discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share,
#                                  min_bin_size=min_bin_size, test_label='out-of-sample', debug=debug)
#     if save_excel:
#         save(data=result_oos, name='test_1_4_oos', path=path)

#     oot = load_dump('validation_params', path)['oot']
#     if oot:
#         X_oot = load_dump('X_oot_proc', path)
#         result_oot = psi_factor_test(X_train, X_oot, discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share,
#                                      min_bin_size=min_bin_size, test_label='out-of-time', debug=debug)
#         if save_excel:
#             save(data=result_oot, name='test_1_4_oot', path=path)
#         return result_oos, result_oot
#     else:
#         return result_oos


# def test_1_8(discr_uniq_val=10,
#              discr_val_share=0.8,
#              min_bin_size=0.05,
#              path='.',
#              save_excel=False,
#              **kwargs):
#     """
#     Calculate PSI for each factor and each class on X_train and X_test. Uses dump data.
#     Executes function psi_factor_test().

#     Two types of factors are defined: continious and discrete.
#     To find out which type the factor is we take discr_uniq_val most common unique values.
#     If number of observations with these values is more than discr_val_share of total sample, factor is labeled discrete.
#     Otherwise it is considered continuous.

#     Type of factor defines binning algorithm used to calculate PSI.

#     :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
#     :param discr_val_share: share of most common values that is enough to define factor as discrete
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :param path: relative or absolute path for output file
#     :param save_excel: whether or not to save excel file (overwrite if already exists)
#     :return: DataFrame with PSI stats by factor
#     """
#     X_train = load_dump('X_train_proc', path)
#     X_test = load_dump('X_test_proc', path)
#     y_train = load_dump('y_train', path)
#     y_test = load_dump('y_test', path)

#     result_oos = psi_factor_classes_test(X_train, X_test, y_train, y_test,
#                                          discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share,
#                                          min_bin_size=min_bin_size, test_label='out-of-sample')
#     if save_excel:
#         save(data=result_oos, name='test_1_8_oos', path=path)

#     oot = load_dump('validation_params', path)['oot']
#     if oot:
#         X_oot = load_dump('X_oot_proc', path)
#         y_oot = load_dump('y_oot', path)
#         result_oot = psi_factor_classes_test(X_train, X_oot, y_train, y_oot,
#                                              discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share,
#                                              min_bin_size=min_bin_size, test_label='out-of-time')
#         if save_excel:
#             save(data=result_oot, name='test_1_8_oot', path=path)
#         return result_oos, result_oot
#     else:
#         return result_oos


# def psi_factor_classes_test(X_train,
#                             X_test,
#                             y_train,
#                             y_test,
#                             discr_uniq_val=10,
#                             discr_val_share=0.8,
#                             min_bin_size=0.05,
#                             test_label='out-of-sample'
#                             ):
#     """
#     Calculate PSI for each factor on X_train and X_test for each label in y_train.
#     Executes psi_factor_test() for each class.

#     :param X_train: first sample
#     :param X_test: second sample
#     :param y_train: first sample labels
#     :param y_test: second sample labels
#     :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
#     :param discr_val_share: share of most common values that is enough to define factor as discrete
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :param test_label: string label for test sample
#     :return: DataFrame with PSI stats by factor by classes (y labels)
#     """
#     feature_types = get_feature_types(
#         X_train, discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share)
#     result = []
#     for class_name in sorted(set(y_train)):
#         X_train_class = X_train[y_train == class_name]
#         X_test_class = X_test[y_test == class_name]
#         class_res = psi_factor_test(X_train_class, X_test_class, min_bin_size=min_bin_size,
#                                     feature_types=feature_types, test_label=test_label)
#         class_res = class_res['Значение PSI']
#         class_res.name = "Class_" + str(class_name) + " PSI"
#         result.append(class_res)
#     result = [pd.Series(feature_types, name="Feature_type").reindex(
#         X_train.keys())] + result
#     result = pd.concat(result, axis=1)
#     result.index = X_train.keys()
#     return result


# def psi_factor_test(X_train,
#                     X_test,
#                     discr_uniq_val=10,
#                     discr_val_share=0.8,
#                     min_bin_size=0.05,
#                     test_label='out-of-sample',
#                     feature_types=None,
#                     debug=False,
#                     ):
#     """
#     Calculate PSI for each factor on X_train and X_test.

#     Two types of factors are defined: continious and discrete.
#     To find out which type is factor we take discr_uniq_val most common unique values.
#     If number of observations with these values is more than discr_val_share of total sample, factor is defined discrete.
#     Otherwise it is considered continuous.

#     Type of factor defines binning algorithm used to calculate PSI.

#     :param X_train: first sample
#     :param X_test: second sample
#     :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
#     :param discr_val_share: share of most common values that is enough to define factor as discrete
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :param test_label: string label for test sample
#     :param feature_types: dict, {factor: 'Discrete'/'Continuous'}. Explicitly pass type of every feature
#     :param debug: turn on debug mode, adding technical info to resulting DataFrame
#     :return: DataFrame with PSI stats by factor
#     """
#     if feature_types is None:
#         feature_types = get_feature_types(
#             X_train, discr_uniq_val=discr_uniq_val, discr_val_share=discr_val_share)
#     result = {}
#     for feature in tqdm(X_train.columns):
#         result[feature] = {}
#         X_train_dropna = X_train[feature][~pd.isna(X_train[feature])]
#         X_test_dropna = X_test[feature][~pd.isna(X_test[feature])]

#         if len(X_train_dropna) == 0 or len(X_test_dropna) == 0:
#             sample = "train" if len(X_train_dropna) == 0 else "test"
#             result[feature] = {'feature_type': f'Every {sample} value is NaN',
#                                'psi': 0, 'bins_cnt': 0, 'bins_train': None, 'bins_test': None}
#             continue

#         if feature_types[feature] == 'Discrete':
#             if len(set(np.unique(X_train_dropna.tolist())).intersection(np.unique(X_test_dropna.tolist()))) == 0:
#                 result[feature] = {'feature_type': 'No label intersection',
#                                    'psi': 0,
#                                    'bins_cnt': 0,
#                                    'bins_train': ' '.join(str(i) for i in np.unique(X_train_dropna.tolist())),
#                                    'bins_test': ' '.join(str(i) for i in np.unique(X_test_dropna.tolist()))}
#             else:
#                 result[feature] = discrete_binning(
#                     X_train_dropna, X_test_dropna, min_bin_size=min_bin_size)
#         else:
#             result[feature] = continuous_binning(
#                 X_train_dropna, X_test_dropna, min_bin_size=min_bin_size)

#     result = pd.DataFrame(
#         result).T[['psi', 'feature_type', 'bins_cnt', 'bins_train', 'bins_test']]
#     result.columns = ['Значение PSI', 'Тип фактора', 'Количество бинов', 'Доли по бинам train',
#                       f'Доли по бинам {test_label}']
#     if debug:
#         return result
#     return result[['Значение PSI']]


# def get_feature_types(df, discr_uniq_val=10, discr_val_share=0.8):
#     """
#     Determine type of every DataFrame column: discrete or continuous.
#     If the number of observations with discr_uniq_val most common unique values
#         is more than discr_val_share of total sample, column is considered discrete.

#     :param df: Input DataFrame
#     :param discr_uniq_val: number of most common unique values to consider when checking if factor is discrete
#     :param discr_val_share: share of most common values that is enough to define factor as discrete
#     :return: dict of {'feature': 'Discrete' or 'Continuous'}
#     """
#     result = {}
#     for feature in df.columns:
#         X_train_dropna = df[feature][~pd.isna(df[feature])]

#         # Get sorted values by counts descending
#         unique_counts = np.unique(
#             X_train_dropna.tolist(), return_counts=True)[1]
#         unique_counts = sorted(unique_counts, reverse=True)

#         if np.sum(unique_counts[:discr_uniq_val]) / len(X_train_dropna) >= discr_val_share \
#                 or df[feature].dtype.name in ["category", "object"]:
#             result[feature] = 'Discrete'
#         else:
#             result[feature] = 'Continuous'
#     return result


# def discrete_binning(X_train, X_test, min_bin_size=0.05):
#     """
#     Make a discrete binning and calculate PSI for two samples: X_train and X_test.

#     :param X_train: 1d array, first sample of values
#     :param X_test: 1d array, second sample of values
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :return: dict with PSI statistics
#     """
#     unique_values, unique_counts = np.unique(
#         X_train.tolist(), return_counts=True)
#     # Sort values by counts descending
#     unique_values = [x for _, x in sorted(
#         zip(unique_counts, unique_values), reverse=True)]
#     unique_counts = sorted(unique_counts, reverse=True)

#     # Merge small bins in train data so that every bin is more than min_bin_size of the X_train without NAs
#     i = 0
#     bin_cnt = 0
#     bins_train = {}
#     total_share = None
#     while i < len(unique_values):
#         total_share = unique_counts[i] / len(X_train)
#         bin_name = 'Bin_' + str(bin_cnt)
#         bin_cnt += 1
#         bins_train[unique_values[i]] = bin_name

#         while total_share < min_bin_size and i < len(unique_values) - 1:
#             i += 1
#             total_share += unique_counts[i] / len(X_train)
#             bins_train[unique_values[i]] = bin_name
#         i += 1

#     # Merge the last bin in train data
#     if total_share < min_bin_size:
#         for value in bins_train:
#             if bins_train[value] == 'Bin_' + str(bin_cnt - 1):
#                 bins_train[value] = 'Bin_' + str(bin_cnt - 2)
#         bin_cnt -= 1

#     # Make dict of {bin_name: value_count} for test data
#     bins_test = np.unique(
#         [bins_train[x] for x in X_test if x in bins_train.keys()], return_counts=True)
#     bins_test = {bin_: count for bin_,
#                  count in zip(bins_test[0], bins_test[1])}

#     # Fill zero bins with {bin_name: 0}
#     for i in range(bin_cnt):
#         if 'Bin_' + str(i) not in bins_test.keys():
#             bins_test['Bin_' + str(i)] = 0

#     # Merge bins that are small in test (less than min_bin_size of the X_test without NAs)
#     i = 0
#     while i < bin_cnt:
#         curr_bin = 'Bin_' + str(i)
#         total_share = bins_test[curr_bin] / len(X_test)
#         while total_share < min_bin_size and i < bin_cnt - 1:
#             i += 1
#             total_share += bins_test['Bin_' + str(i)] / len(X_test)
#             for value in bins_train:
#                 if bins_train[value] == 'Bin_' + str(i):
#                     bins_train[value] = curr_bin
#         i += 1
#     bin_cnt = len(set(bins_train.values()))

#     # Merge the last bin for test data
#     if total_share < min_bin_size and len(set(bins_train.values())) > 1:
#         bin_nums = [int(i[4:]) for i in set(bins_train.values())]
#         bin_nums = sorted(bin_nums, reverse=True)
#         for value in bins_train:
#             if bins_train[value] == 'Bin_' + str(bin_nums[0]):
#                 bins_train[value] = 'Bin_' + str(bin_nums[1])
#         bin_cnt -= 1

#     # Recount bin shares according to final binning
#     final_bins_train = np.unique(
#         [bins_train[x] for x in X_train if x in bins_train.keys()], return_counts=True)
#     final_bins_train = {
#         bin_: count / len(X_train) for bin_, count in zip(final_bins_train[0], final_bins_train[1])}

#     final_bins_test = np.unique(
#         [bins_train[x] for x in X_test if x in bins_train.keys()], return_counts=True)
#     final_bins_test = {
#         bin_: count / len(X_test) for bin_, count in zip(final_bins_test[0], final_bins_test[1])}

#     # Calculate psi
#     psi = 0
#     for bin_ in final_bins_train:
#         train_share = final_bins_train[bin_]
#         test_share = final_bins_test[bin_]
#         psi += (train_share - test_share) * math.log(train_share / test_share)

#     result = {'psi': psi,
#               'bins_cnt': bin_cnt,
#               'bins_train': ' '.join(['%.3f' % val for val in final_bins_train.values()]),
#               'bins_test': ' '.join(['%.3f' % val for val in final_bins_test.values()]),
#               'feature_type': 'Discrete'}

#     return result


# def continuous_binning(X_train, X_test, min_bin_size=0.05):
#     """
#     Make a continuous binning and calculate PSI for two samples: X_train and X_test.

#     :param X_train: 1d array, first sample of values
#     :param X_test: 1d array, second sample of values
#     :param min_bin_size: minimal size of bin in proportion to total number of observations in sample
#     :return: dict with PSI statistics
#     """
#     bin_points = [-np.inf, np.inf]
#     binning_possible = True
#     while binning_possible:
#         binning_possible = False

#         # Try to split each existing bin on np.median(X_train)
#         for j in range(len(bin_points) - 1):
#             X_train_bin = X_train[(X_train > bin_points[j]) & (
#                 X_train <= bin_points[j + 1])]

#             if len(X_train_bin) / 2 >= min_bin_size * len(X_train):
#                 median = np.median(X_train_bin)
#                 test_left = ((X_test > bin_points[j]) & (
#                     X_test <= median)).sum()
#                 test_right = ((X_test > median) & (
#                     X_test <= bin_points[j + 1])).sum()
#                 if test_left >= min_bin_size * len(X_test) and test_right >= min_bin_size * len(X_test):
#                     binning_possible = True
#                     bin_points.append(median)
#                     bin_points.sort()
#                     break

#     # Calculate psi after final binning
#     psi = 0
#     bins_train = {}
#     bins_test = {}
#     for i in range(len(bin_points) - 1):
#         train_share = ((X_train > bin_points[i]) & (
#             X_train <= bin_points[i + 1])).sum() / len(X_train)
#         test_share = ((X_test > bin_points[i]) & (
#             X_test <= bin_points[i + 1])).sum() / len(X_test)
#         bins_train['Bin_' + str(i)] = train_share
#         bins_test['Bin_' + str(i)] = test_share
#         psi += (train_share - test_share) * math.log(train_share / test_share)

#     result = {'psi': psi,
#               'bins_cnt': len(bin_points) - 1,
#               'bins_train': ' '.join(['%.3f' % val for val in bins_train.values()]),
#               'bins_test': ' '.join(['%.3f' % val for val in bins_test.values()]),
#               'feature_type': 'Continuous'}

#     return result

# # test_1_4(path='Validation_regression')
# # test_1_8(path='Validation_multiclass')
