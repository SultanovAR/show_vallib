import numpy as np
import typing as tp
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from .base import BaseSampler


class BinarySampler(BaseSampler):
    def __init__(
        self,
        train: dict,
        oos: dict,
        oot: dict=None,
        bootstrap=False,
        **kwargs
    ):
        """
        train: {'X': , 'y_true': , 'y_pred':}
        todo
        """
        super().__init__(train, oos, oot, bootstrap, **kwargs)
        
        self.check()
        
    def check(self):
        pass
    
    
    def set_seed(self, seed: int):
        
        generator = np.random.default_rng(seed=seed)
        size_of_train = len(self.source_train)
        size_of_oos = len(self.source_oos)
        
        if self.bootstrap:
            self.index = {'train': None, 'oos': generator.integers(0, size_of_oos)}
        else:
            permutation = generator.permutation(np.arange(len(self.source_train)+ len(self.source_oos)))
            self.index = {'train': permutation[:size_of_train], 'oos': permutation[size_of_train:]}
            
            
    @property
    def train(self):
        if self.source_state or self.bootstrap:
            return self.source_train
        
        else:
            if isinstance(self.source_train['X'], pd.DataFrame):
                X_concat = pd.concat([self.source_train['X'], self.source_oos['X']])
                ans = X_concat[self.index['train']]
            # todo 
            
    # todo @property oos, oot
