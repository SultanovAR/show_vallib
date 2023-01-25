import numpy as np
import typing as tp
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sbe_vallib.validation.sampler import BaseSampler


class BinarySampler(BaseSampler):
    def __init__(
        self,
        train: dict,
        oos: dict,
        oot: dict = None,
        bootstrap: bool = False,
        stratify: bool = True,
        **kwargs
    ):
        """
        :param train: dict: Store the training data
        :param oos: dict: Store the oos data
        :param oot: dict: Specify a test set that is not part of the training set
        :param bootstrap: bool: Indicate whether the model is trained on bootstrapped data
        :param **kwargs: Pass a variable number of keyword arguments to a function
        :return: None
        """
        super().__init__(train, oos, oot, bootstrap, **kwargs)
        self.stratify = stratify
        self.index = dict()

    def _is_pandas(self, data):
        """A function to deal with pd.DataFrame and np.array in the same manner"""
        return hasattr(data, 'iloc')

    def _concat(self, head, tail):
        """A function to deal with pd.DataFrame and np.array in the same manner"""
        if self._is_pandas(head) and self._is_pandas(tail):
            return pd.concat([head, tail])
        return np.concatenate((head, tail))

    def _get_index(self, data, index):
        """A function to deal with pd.DataFrame and np.array in the same manner"""
        if self._is_pandas(data):
            return data.iloc[index]
        return data[index]

    def set_seed(self, seed: int):
        """
        The set_seed function is used to set the seed for the random number generator.
        The purpose of this function is to ensure that we can reproduce our results by
        ensuring that we always use the same seed and permutation when bootstrapping.

        :param seed: int: Set the seed for the random number generator
        :return: The value of the seed
        """
        self.source_state = False
        generator = np.random.default_rng(seed=seed)
        size_of_train = len(self.source_train["X"])
        size_of_oos = len(self.source_oos["X"])

        if self.bootstrap:
            self.index = {
                "train": None,
                "oos": generator.integers(0, size_of_oos, size_of_oos),
            }
        else:
            target_for_stratify = None
            if self.stratify:
                target_for_stratify = self._concat(
                    self.source_train["y_true"], self.source_oos['y_true']
                )
            self.index['train'], self.index['oos'] = train_test_split(np.arange(size_of_train + size_of_oos),
                                                                      test_size=size_of_oos,
                                                                      random_state=seed,
                                                                      shuffle=True,
                                                                      stratify=target_for_stratify)

    @property
    def train(self):
        if self.source_state or self.bootstrap:
            return self.source_train

        result = {}
        for key in self.source_train:
            concated = self._concat(
                self.source_train[key], self.source_oos[key])
            result[key] = self._get_index(concated, self.index["train"])
        return result

    @property
    def oos(self):
        if self.source_state:
            return self.source_oos

        result = {}
        if self.bootstrap:
            for key in self.source_oos:
                result[key] = self._get_index(
                    self.source_oos[key], self.index['oos'])
        else:
            for key in self.source_train:
                concated = self._concat(
                    self.source_train[key], self.source_oos[key])
                result[key] = self._get_index(concated, self.index["oos"])
        return result

    @property
    def oot(self):
        return self.source_oot
