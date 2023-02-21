from abc import ABC, abstractmethod


class BaseScorer(ABC):
    def __init__(self, *args, **kwargs):
        pass
        # , metrics: dict, custom_metrics={}, **kwargs):
        # self.metrics = metrics
        # self.metrics.update(custom_metrics)
        # self.metrics = {key: METRICS[key] for key in metrics}

    # def _init_scorers(self):
    #     scorers = dict()
    #     for metric in self.metrics:
    #         scorers[metric] = self.init_scorer(self.metrics[metric])

    # @abstractmethod
    # def init_scorer(self, metric: callable):
    #     raise NotImplementedError

    @abstractmethod
    def calc_metrics(self, *args, **kwargs):
        raise NotImplementedError
