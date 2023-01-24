import numpy as np

from sbe_vallib.validation.scorer.base import BaseScorer
from sbe_vallib.validation.utils.metrics import (
    BINARY_METRICS,
    REGRESSION_METRICS,
    MULTICLASS_METRICS,
)


class BinaryScorer(BaseScorer):
    def __init__(self, metrics=BINARY_METRICS, cutoff=0.5, custom_scorers={}, **kwargs):
        super().__init__(metrics, custom_scorers, **kwargs)
        self.cutoff = cutoff

    def score(self, y_true, y_proba):
        answer = {}
        for metric_name in self.base_metrics:
            if self.base_metrics[metric_name]["use_probas"]:
                answer[metric_name] = self.base_metrics[metric_name]["callable"](
                    y_true, y_proba[:, 1]
                )
            else:
                answer[metric_name] = self.base_metrics[metric_name]["callable"](
                    y_true, np.array(y_proba[:, 1] > self.cutoff, dtype=int)
                )

        return answer


class RegressionScorer(BaseScorer):
    def __init__(self, metrics=REGRESSION_METRICS, custom_scorers={}, **kwargs):
        super().__init__(metrics, custom_scorers, **kwargs)

    def score(self, y_true, y_preds):
        answer = {}
        for metric_name in self.base_metrics:
            answer[metric_name] = self.base_metrics[metric_name]["callable"](
                y_true, y_preds
            )

        return answer


class MulticlassScorer(BaseScorer):
    def __init__(self, metrics=MULTICLASS_METRICS, custom_scorers=..., **kwargs):
        super().__init__(metrics, custom_scorers, **kwargs)

    def score(self, y_true, y_preds):
        answer = {}
        for metric_name in self.base_metrics:
            answer[metric_name] = self.base_metrics[metric_name]["callable"](
                y_true, y_preds, average=self.base_metrics[metric_name]["average"]
            )

        return answer


class MultilabelScorer(BaseScorer):
    def __init__(self, metrics, custom_scorers=..., **kwargs):
        super().__init__(metrics, custom_scorers, **kwargs)

        raise NotImplementedError
