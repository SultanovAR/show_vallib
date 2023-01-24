from abc import ABC, abstractmethod
# from sbe_vallib.metrics import METRICS
from sklearn.metrics import f1_score

METRICS = {'f1': f1_score}


class BaseScorer(ABC):
    def __init__(self, metrics, custom_scorers={}, **kwargs):
        self.base_metrics = metrics
        self.custom_scorers = custom_scorers
        # self.metrics = {key: METRICS[key] for key in metrics}
    

    # def _init_scorers(self):
    #     scorers = dict()
    #     for metric in self.metrics:
    #         scorers[metric] = self.init_scorer(self.metrics[metric])
    
    # @abstractmethod
    # def init_scorer(self, metric: callable):
    #     raise NotImplementedError
    
    @abstractmethod
    def score(self, *args, **kwargs):
        raise NotImplementedError
