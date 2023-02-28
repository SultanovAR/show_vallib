import numpy as np

from sbe_vallib.scorer.base import BaseScorer
from sbe_vallib.utils.metrics import (
    BINARY_METRICS,
    REGRESSION_METRICS,
    MULTICLASS_METRICS,
)

# добавить


class BinaryScorer(BaseScorer):
    def __init__(self, metrics=BINARY_METRICS, cutoff=0.5, custom_metrics={}, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.metrics.update(custom_metrics)
        self.cutoff = cutoff

    def calc_metrics(self, y_true=None, y_proba=None,
                     model=None, sampler=None, data_type=None, **kwargs):
        if y_proba is None:
            y_proba = model.predict_proba(getattr(sampler, data_type)['X'])
        if y_true is None:
            y_true = getattr(sampler, data_type)['y_true']

        answer = {}
        for metric_name in self.metrics:
            if self.metrics[metric_name]["use_probas"]:
                answer[metric_name] = self.metrics[metric_name]["callable"](
                    y_true, y_proba[:, 1]
                )
            else:
                answer[metric_name] = self.metrics[metric_name]["callable"](
                    y_true, np.array(y_proba[:, 1] > self.cutoff, dtype=int)
                )

        return answer


# class RegressionScorer(BaseScorer):
#     def __init__(self, metrics=REGRESSION_METRICS, custom_scorers={}, **kwargs):
#         super().__init__(metrics, custom_scorers, **kwargs)

#     def calc_metrics(self, y_true, y_preds):
#         answer = {}
#         for metric_name in self.metrics:
#             answer[metric_name] = self.metrics[metric_name]["callable"](
#                 y_true, y_preds
#             )

#         return answer


# class MulticlassScorer(BaseScorer):
#     def __init__(self, metrics=MULTICLASS_METRICS, custom_scorers=..., **kwargs):
#         super().__init__(metrics, custom_scorers, **kwargs)

#     def calc_metrics(self, y_true, y_preds):
#         answer = {}
#         for metric_name in self.metrics:
#             answer[metric_name] = self.metrics[metric_name]["callable"](
#                 y_true, y_preds, average=self.metrics[metric_name]["average"]
#             )

#         return answer


# class MultilabelScorer(BaseScorer):
#     def __init__(self, metrics, custom_scorers=..., **kwargs):
#         super().__init__(metrics, custom_scorers, **kwargs)

#         raise NotImplementedError
