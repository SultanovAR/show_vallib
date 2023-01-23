from sbe_vallib.validation.scorer.base import BaseScorer


class BinaryScorer(BaseScorer): # для разных метрик needs_proba + катоф
    def __init__(self, metrics, custom_metrics={}, **kwargs):
        super().__init__(metrics, custom_metrics, **kwargs)

    def score(self, model, data):

        answer = {}
        if self.calc_every_class:
            pass
        else:
            for key in self.metrics.keys():
                answer[key] = self.metrics[key](y_target, y_pred)

        return answer
