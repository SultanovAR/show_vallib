from abc import ABC, abstractmethod


class BaseScorer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calc_metrics(self, model=None, sampler=None, data_type=None, *args, **kwargs):
        raise NotImplementedError
