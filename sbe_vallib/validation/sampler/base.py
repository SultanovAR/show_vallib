import pandas as pd
from abc import ABC, abstractmethod
import typing as tp


class BaseSampler(ABC):
    def __init__(
        self,
        train,
        oos: tp.Optional[dict],
        oot: tp.Optional[dict],
        # bootstrap: bool = False,
        **kwargs
    ):

        self.source_train = train
        self.source_oos = oos
        self.source_oot = oot
        self.source_state = True

        # self.bootstrap = bootstrap

    def reset(self):
        self.source_state = True

    @property
    def train(self):
        return self.source_train

    @property
    def oos(self):
        return self.source_oos

    @property
    def oot(self):
        return self.source_oot

    @abstractmethod
    def set_seed(self):
        self.source_state = False
        pass
