from typing import Union, List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_array
import physt

import sbe_vallib.validation.utils.pd_np_interface as interface


class Quantization(BaseEstimator, TransformerMixin):

    def __init__(self,
                 merge_quantile=0.05,
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
        bins = np.unique(data)
        initial_width = np.min(np.abs(bins[:-1] - bins[1:])) / 2
        bins = np.concatenate(
            (bins - initial_width, bins + initial_width), axis=0)
        bins = np.unique(np.round(bins, self.rounding_precision))

        hist = physt.h1(data, bins=bins)
        hist = hist.merge_bins(min_frequency=len(data) * self.merge_quantile)
        bins = np.concatenate([[-np.inf], hist.edges, [np.inf]])
        return bins

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None, columns: List = None):
        if columns is None:
            columns = interface.all_columns(X)

        data = np.array(interface.get_columns(X, columns))
        self._check_nan(data)
        check_array(X, force_all_finite=False)

        for i, col in enumerate(range(data.shape[1])):
            self.bins[col] = self.merging_binning(data[:, i])
            # edges = physt.histogram(data[:, i], 'knuth').edges
            # self.bins[col] = np.concatenate([[-np.inf], edges, [np.inf]])
        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]):
        quantized = deepcopy(data)
        check_array(quantized, force_all_finite=False)
        for col in self.bins:
            samples = interface.get_columns(quantized, col)
            self._check_nan(samples)
            binned_data = np.digitize(samples, bins=self.bins[col])
            interface.set_column(quantized, binned_data, col)
        return quantized
