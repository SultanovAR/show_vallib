import numpy as np
import typing as tp
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sbe_vallib.validation.sampler import BaseSampler

from sbe_vallib.validation.utils import concat, get_index


class NerSampler(BaseSampler):
    def __init__(
        self, train: dict, oos: dict, oot: dict = None, stratify: bool = True, **kwargs
    ):
        """
        for example:
        tr
        train['true'][i] = {'id': '2',
                    'tokens': ['AL-AIN', ',', 'United', 'Arab', 'Emirates', '1996-12-06'],
                    'ner_tags': [5, 0, 5, 6, 6, 0]}

        train['pred'][i] = {'id': '2'
                    'ner_tags': [5, 0, 5, 6, 6, 0]}

        """

        super().__init__(train, oos, oot, **kwargs)
        self.stratify = stratify
        self.index = dict()

    def set_seed(self, seed: int, bootstrap: bool = False):
        
        self.bootstrap = bootstrap
        self.source_state = False
        generator = np.random.default_rng(seed=seed)
        
        size_of_train = len(self.source_train['true'])
        size_of_oos = len(self.source_oos['true'])

        if bootstrap:
            self.index = {
                "train": None,
                "oos": generator.integers(0, size_of_oos, size_of_oos),
            }
        else:
            self.index = {"train": None, "oos": None}
            self.index["train"], self.index["oos"] = train_test_split(
                np.arange(size_of_train + size_of_oos),
                test_size=size_of_oos,
                random_state=seed,
                shuffle=True,
            )

    @property
    def train(self):
        if self.source_state or self.bootstrap:
            return self.source_train

        result = {}
        for key in self.source_train:
            concated = concat([self.source_train[key], self.source_oos[key]])
            result[key] = get_index(concated, self.index["train"])
        return result

    @property
    def oos(self):
        if self.source_state:
            return self.source_oos

        result = {}
        if self.bootstrap:
            for key in self.source_oos:
                print(key)
                result[key] = get_index(self.source_oos[key], self.index["oos"])
        else:
            for key in self.source_train:
                concated = concat([self.source_train[key], self.source_oos[key]])
                result[key] = get_index(concated, self.index["oos"])
        return result

    @property
    def oot(self):
        return self.source_oot
