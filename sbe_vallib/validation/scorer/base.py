from abc import ABC, abstractmethod
# from sbe_vallib.metrics import METRICS
from sklearn.metrics import f1_score

METRICS = {'f1': f1_score}


class BaseScorer(ABC):
    def __init__(self, metrics, *, custom_metrics={}, calc_every_class=False, **params):
        """
        Объект, который считает указанные метрики

        @metrics: список метрик
        @custom_metrics: словарь с кастомными метриками
        @calc_every_class: считать метрику для каждого класса

        return:
        """
        self.metrics = {key: METRICS[key] for key in metrics}
        self.metrics.update(custom_metrics)

        self.calc_every_class = calc_every_class

    @abstractmethod
    def score(self, y_target, y_pred):
        raise NotImplementedError
