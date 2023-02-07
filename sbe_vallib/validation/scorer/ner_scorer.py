import typing as tp
from collections import defaultdict

from nervaluate import Evaluator

from sbe_vallib.validation.scorer.base import BaseScorer


class NerScorer(BaseScorer):
    def __init__(self, metrics: tp.Dict,
                 custom_metrics={}):
        super().__init__(metrics, custom_metrics)

    def ner_metrics(self, y_true, y_pred, classes, **kwargs):
        answer = defaultdict(dict)
        evaluator = Evaluator(y_true, y_pred, classes, loader='list')
        ner_metrics_macro, ner_metrics_by_tag = evaluator.evaluate()

        for schema in ['ent_type', 'partial', 'strict', 'exact']:
            for tag in classes + ['macro']:
                if tag == 'macro':
                    answer[f'f1_{schema}'].update(
                        {tag: ner_metrics_macro[schema]['f1']})
                else:
                    answer[f'f1_{schema}'].update(
                        {tag: ner_metrics_by_tag[tag][schema]['f1']})
        return answer

    def score(self, y_true, y_pred, **kwargs):
        answer = {}
        for metric_name in self.metrics:
            answer[metric_name] = self.metrics[metric_name]["callable"](
                y_true, y_pred, **kwargs
            )
        return answer
