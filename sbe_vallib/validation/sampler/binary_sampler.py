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
        self.check()

    def check(self):
        """
        The check function is used to determine whether the input data is a DataFrame or an array.
        The check function returns True if the input data is a DataFrame and False if it's an array.

        :param self: Access the attributes and methods of the class in python
        :return: A boolean value indicating whether the data is a pandas dataframe
        """
        if isinstance(self.source_train["X"], pd.DataFrame):
            self.is_data_frame = True
        elif isinstance(self.source_train["X"], np.ndarray):
            self.is_data_frame = False

        self.concat = {True: pd.concat, False: np.concatenate}

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
            permutation = generator.permutation(np.arange(size_of_train + size_of_oos))
            self.index = {
                "train": permutation[:size_of_train],
                "oos": permutation[size_of_train:],
            }


    @property
    def train(self):
        if self.source_state or self.bootstrap:
            return self.source_train
        else:
            X_concat = self.concat[self.is_data_frame](
                [self.source_train["X"], self.source_oos["X"]]
            )
            y_true_concat = self.concat[self.is_data_frame](
                [self.source_train["y_true"], self.source_oos["y_true"]]
            )
            y_pred_concat = np.concatenate(
                [self.source_train["y_pred"], self.source_oos["y_pred"]]
            ) # через трейнтестсплит/возможно отсутствие y_pred / обложить тестами

            if self.is_data_frame:
                return {
                    "X": X_concat.loc[self.index["train"]],
                    "y_true": y_true_concat.loc[self.index["train"]],
                    "y_pred": y_pred_concat[self.index["train"]],
                }
            else:
                return {
                    "X": X_concat[self.index["train"]],
                    "y_true": y_true_concat[self.index["train"]],
                    "y_pred": y_pred_concat[self.index["train"]],
                }

    @property
    def oos(self):
        if self.source_state:
            return self.source_oos
        elif self.bootstrap:
            if self.is_data_frame:
                X = self.source_oos["X"].iloc[self.index["oos"]]
                y_true = self.source_oos["y_true"].iloc[self.index["oos"]]
            else:
                X = self.source_oos["X"][self.index["train"]]
                y_true = self.source_oos["y_true"][self.index["oos"]]

            y_pred = self.source_oos["y_pred"][self.index["oos"]]
            return {"X": X, "y_true": y_true, "y_pred": y_pred}
        else:
            X_concat = self.concat[self.is_data_frame](
                [self.source_train["X"], self.source_oos["X"]]
            )
            y_true_concat = self.concat[self.is_data_frame](
                [self.source_train["y_true"], self.source_oos["y_true"]]
            )
            y_pred_concat = np.concatenate(
                [self.source_train["y_pred"], self.source_oos["y_pred"]]
            )
            if self.is_data_frame:
                return {
                    "X": X_concat.loc[self.index["oos"]],
                    "y_true": y_true_concat.loc[self.index["oos"]],
                    "y_pred": y_pred_concat[self.index["oos"]],
                }
            else:
                return {
                    "X": X_concat[self.index["oos"]],
                    "y_true": y_true_concat[self.index["oos"]],
                    "y_pred": y_pred_concat[self.index["oos"]],
                }

    @property
    def oot(self):
        return self.source_oot
