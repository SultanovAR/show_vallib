from abc import ABC, abstractmethod
# from sbe_vallib.metrics import METRICS
from sklearn.metrics import f1_score

METRICS = {'f1': f1_score}


class BaseScorer(ABC):
    def __init__(self, metrics, custom_metrics={}, **kwargs):
        self.metrics = {key: METRICS[key] for key in metrics}
        self.metrics.update(custom_metrics)

    @abstractmethod
    def score(self, model, data):
        raise NotImplementedError
