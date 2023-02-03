from nervaluate import Evaluator

from sbe_vallib.validation.scorer.base import BaseScorer


class NerScorer(BaseScorer):
    def __init__(self, metrics: dict,
                 custom_metrics={},
                 is_calc_ner_metrics=True,
                 tags=['PER', 'MISC', 'LOC', 'ORG'],
                 **kwargs, ):
        super().__init__(metrics, custom_metrics)
        self.tags = tags
        self.is_calc_ner_metrics = is_calc_ner_metrics

    def ner_metrics(self, y_true, y_pred, **kwargs):
        answer = {}
        evaluator = Evaluator(y_true, y_pred, tags=self.tags, loader='list')
        ner_metrics, ner_metrics_by_tag = evaluator.evaluate()

        for schema in ['ent_type', 'partial', 'strict', 'exact']:
            for metric in ['f1', 'recall', 'precision']:
                answer[f"{schema}_{metric}"] = ner_metrics[schema][metric]
                for tag in self.tags:
                    answer[f"{schema}_{metric}_by_{tag}"] = ner_metrics_by_tag[tag][schema][metric]
        return answer

    def score(self, y_true, y_pred, **kwargs):
        answer = {}
        for metric_name in self.metrics:
            answer[metric_name] = self.metrics[metric_name]["callable"](
                y_true, y_pred
            )
        if self.is_calc_ner_metrics:
            answer.update(self.ner_metrics(y_true, y_pred))
        return answer
