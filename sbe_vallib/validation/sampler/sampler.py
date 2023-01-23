import numpy as np

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from .base import BaseSampler


class TabularSampler(BaseSampler):
    def __init__(
        self,
        X_train,
        y_train,
        X_oos,
        y_oos,
        *,
        X_oot=None,
        y_oot=None,
        y_pred_train=None,
        y_pred_oos=None,
        y_pred_oot=None,
        bootstrap_conf_int=False,
        **kwargs
    ):

        super().__init__(X_train, y_train, X_oos, y_oos, **kwargs)

        self.X_oot = X_oot
        self.y_oot = y_oot

        self.y_pred_train = y_pred_train
        self.y_pred_oos = y_pred_oos
        self.y_pred_oot = y_pred_oot

        self.bootstrap_conf_int = bootstrap_conf_int

        self.samplers = {
            "train": (X_train, y_train, y_pred_train),
            "oos": (X_oos, y_oos, y_pred_oos),
            "oot": (X_oot, y_oot, y_pred_oot),
        }

    def boots(self):
        return self.bootstrap_conf_int

    def keys(self):
        return self.samplers.keys()

    def __getitem__(self, key):
        return self.samplers[key]

    def sample(self, num):
        """
        Метод генерации подвыборок

        @num: номер псевдослучайной подпоследовательности
        """
        if self.bootstrap:
            X_train_boots, y_train_boots = self.boots_sample(
                self.X_train, self.y_train, self.len_train, num
            )

            X_oos_boots, y_oos_boots = self.boots_sample(
                self.X_oos, self.y_oos, self.len_oos, num
            )

            return X_train_boots, X_oos_boots, y_train_boots, y_oos_boots

        else:
            return train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=num
            )

    @staticmethod
    def boots_sample(X, y, len_, num):
        np.random.seed(num)
        idx = np.random.randint(0, len_, size=(len_,))

        if isinstance(X, DataFrame):
            X_boots = X.reset_index(drop=True).iloc[idx]
        elif isinstance(X, np.ndarray):
            X_boots = X[idx]
        else:
            raise TypeError("Неподдерживаемый формат X")

        y_boots = y[idx]

        return X_boots, y_boots
