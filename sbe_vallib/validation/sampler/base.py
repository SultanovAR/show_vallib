import pandas as pd
from abc import ABC, abstractmethod
import typing as tp

class BaseSampler(ABC):
    def __init__(
        self, train: tp.Optional(), oos: tp.Optional(), oot: tp.Optional(), bootstrap: bool=False, **kwargs
    ):

        self.source_train = train
        self.source_oos = oos
        self.source_oot = oot
        self.source_state = True
                
        self.bootstrap = bootstrap
        
    def reset(self):
        self.source_state = True
    
    @property
    def train(self):
        return self.source_train
    
    # oos oot todo
    
    @abstractmethod
    def set_seed(self):
        self.source_state = False
        pass
    
    
    
    
    
    
