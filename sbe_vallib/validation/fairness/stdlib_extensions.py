from typing import Union, List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import physt

from utils import get_columns, set_column, all_columns, copy_docstring_from


class Interval(pd.Interval):
    
    @copy_docstring_from(pd.Interval.__init__)
    def __init__(self, left, right, closed='neither'):
        super().__init__(left, right, closed)
        
    def __and__(self, other: pd.Interval):
        if not self.overlaps(other):
            return pd.Interval(0, 0, closed='neither') #empty set
        left = max(self.left, other.left)
        right = min(self.right, other.right)
        left_in = (left in self) and (left in other)
        right_in = (right in self) and (right in other)
        closed = 'neither'
        if left_in and right_in:
            closed = 'both'
        if left_in and not right_in:
            closed = 'left'
        if not left_in and right_in:
            closed = 'right'
        return pd.Interval(left, right, closed)        


class CatBoostClassifierTrainTestSplit(CatBoostClassifier):
    
    def __init__(self, test_size=0.2, *args, **argv):
        self.test_size = test_size
        super().__init__(*args, **argv)
        
    def fit(self, x, y, **argv):
        train_x, eval_x, train_y, eval_y = train_test_split(x, y, test_size=self.test_size)
        return super().fit(X=train_x, y=train_y, eval_set=(eval_x, eval_y))


class Quantization(BaseEstimator, TransformerMixin):
    
    def __init__(self, merge_quantile=0.05,
                 rounding_precision=5):
        super().__init__()
        self.merge_quantile = merge_quantile
        self.rounding_precision = rounding_precision
        self.bins = dict()
    
    def _check_nan(self, x):
        if np.isnan(x).sum() > 0:
            raise ValueError("""Data in 'columns' of 'x' must not contain nan values.
            You can replace it with np.inf or something else""")
            
    def merging_binning(self, data: np.ndarray):
        bins = np.unique(data[~np.isnan(data)])
        initial_width = np.min(np.abs(bins[:-1] - bins[1:])) / 2
        bins = np.concatenate((bins - initial_width, bins + initial_width), axis=0)
        bins = np.unique(np.round(bins, self.rounding_precision))
        
        hist = physt.h1(data, bins=bins)
        hist = hist.merge_bins(min_frequency=len(data) * self.merge_quantile)
        return hist.edges
        
    def fit(self, x: Union[np.ndarray, pd.DataFrame], y=None, columns: List=None):
        check_array(x, force_all_finite=False)
        
        if columns is None:
            columns = all_columns(x)
        data = np.array(get_columns(x, columns))
        self._check_nan(data)
        
        for i, col in enumerate(columns):
            self.bins[col] = self.merging_binning(data[:, i])
        return self
        
    def transform(self, data: Union[np.ndarray, pd.DataFrame]):
        x = deepcopy(data)
        check_array(x, force_all_finite=False)
        for col in self.bins:
            samples = get_columns(x, col)
            self._check_nan(samples)
            binned_data = np.digitize(samples, bins=self.bins[col])
            set_column(x, binned_data, col)
        return x
        
        

        
        
        
        