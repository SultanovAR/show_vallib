from typing import Union, List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array
from physt.binnings import scott_binning

import sbe_vallib.utils.pd_np_interface as interface


class Quantizer(BaseEstimator, TransformerMixin):
    TRASH_CAT_BUCKET = '__other'

    def __init__(self,
                 less_count_is_cat: int = 20,
                 drop_cat_freq_less: float = 0.05,
                 ):
        self.less_count_is_cat = less_count_is_cat
        self.drop_cat_freq_less = drop_cat_freq_less
        self.bins = {}
        self.columns = None
        self.cat_columns = None

    def fit(self, X, y=None, columns=None):
        if columns is None:
            self.columns = interface.all_columns(X)
        else:
            self.columns = columns
        self.cat_columns = self.get_cat_features(
            interface.get_columns(X, self.columns), self.columns)

        for column in self.columns:
            data_column = np.ravel(interface.get_columns(X, [column]))
            if column in self.cat_columns:
                self.bins[column] = self.get_good_represented_cats(data_column)
            else:
                self.bins[column] = scott_binning(data_column).numpy_bins

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

    def get_cat_features(self, data, columns):
        df = pd.DataFrame(data, columns=columns)
        cat_features = df.select_dtypes(
            include=['object', 'category']).columns.to_list()
        low_unique = df.columns[df.nunique(
        ) <= self.less_count_is_cat].to_list()
        cat_features = list(set(cat_features) | set(low_unique))
        return cat_features

    def cat_digitize(self, series, bins):
        return list(map(lambda x: x if x in bins else self.TRASH_CAT_BUCKET, series))
