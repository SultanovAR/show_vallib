from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from physt.binnings import scott_binning
from physt.facade import histogram

import sbe_vallib.utils.pd_np_interface as interface
from sbe_vallib.utils.cat_features import get_cat_features


class Quantizer(BaseEstimator, TransformerMixin):
    TRASH_CAT_BUCKET = '__other'

    def __init__(self,
                 less_count_is_cat: int = 20,
                 drop_cat_freq_less: float = 0.05,
                 min_freq_for_bin: int = 35):
        self.less_count_is_cat = less_count_is_cat
        self.drop_cat_freq_less = drop_cat_freq_less
        self.bins = {}
        self.columns = None
        self.cat_columns = None
        self.min_freq_for_bin = min_freq_for_bin

    def fit(self, X, columns=None, **kwargs):
        self.columns = interface.all_columns(X) if columns is None else columns
        self.cat_columns = get_cat_features(
            interface.get_columns(X, self.columns), self.less_count_is_cat)

        for column in self.columns:
            data_column = np.ravel(interface.get_columns(X, [column]))
            if column in self.cat_columns:
                self.bins[column] = self.get_good_represented_cats(data_column)
            else:
                bins = scott_binning(data_column).numpy_bins
                physt_hist = histogram(data_column, bins=bins).merge_bins(
                    min_frequency=self.min_freq_for_bin)
                bins = physt_hist.numpy_bins
                bins[0], bins[-1] = -np.inf, np.inf
                self.bins[column] = bins
        return self

    def transform(self, X):
        _X = deepcopy(X)
        for column in self.columns:
            data_column = np.ravel(interface.get_columns(_X, [column]))
            if column in self.cat_columns:
                binned = self.cat_digitize(data_column, self.bins[column])
                interface.set_column(_X, binned, column)
            else:
                binned = np.digitize(data_column, bins=self.bins[column])
                interface.set_column(_X, binned, column)
        return _X

    def get_bins(self, column):
        if column not in self.bins:
            raise ValueError(
                'Quantizer is not fitted on this column, check parameter "column" in fit method')

        if column in self.cat_columns:
            return self.bins[column] + [self.TRASH_CAT_BUCKET]
        else:
            return self.bins[column]

    def get_good_represented_cats(self, series):
        counts = pd.value_counts(series).reset_index(
            name='count').sort_values(by='count')
        counts['freq'] = counts['count'] / counts['count'].sum()
        remain_cats = (
            counts[counts['freq'] >= self.drop_cat_freq_less]['index']).to_list()
        return remain_cats

    def cat_digitize(self, series, bins):
        return list(map(lambda x: x if x in bins else self.TRASH_CAT_BUCKET, series))
