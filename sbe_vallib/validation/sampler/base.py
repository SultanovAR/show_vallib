import pandas as pd
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(
        self, X_train, y_train, X_oos, y_oos, *, bootstrap_conf_int=False, **kwargs
    ):

        """
        Объект для семплирования выборок

        @X_train: обучающая выборка
        @y_train: значения целевой переменной для обучения
        @X_oos: тестовая выборка
        @y_oos: значения целевой переменной для тестирования
        @bootstrap_conf_int: флаг метода получения подвыборок

        return Object
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_oos = X_oos
        self.y_oos = y_oos

        self.len_train = len(X_train)
        self.len_oos = len(X_oos)

        self.test_size = self.len_oos / (self.len_oos + self.len_train)

        self.X = pd.concat([X_train, X_oos])
        self.y = pd.concat([y_train, y_oos])

        self.bootstrap = bootstrap_conf_int

    @abstractmethod
    def sample(self, num):
        raise NotImplementedError
