from sbe_vallib.validation.scorer.base import BaseScorer
from sbe_vallib.util.metrics import BINARY_METRICS
# from sklearn.metrics import  описать словарь метрик (стандартных) параметр giin/f1 в валидоре

# BINARY_METRICS = [
#     {'metric_key': 'f1',
#      'metric_callable': f1_score,
#      'use_proba': True,
#      },
#     {
#         'metric_key': 'gini',
#         'merti'
#     }
# ]


class BinaryScorer(BaseScorer): # для разных метрик needs_proba + катоф
    def __init__(self, metrics, cutoff=0.5, custom_scorers={}, **kwargs):
        super().__init__(metrics, custom_scorers, **kwargs)
        self.cutoff = cutoff
        
    
    def init_scorer(self, metric: callable):
        pass

    def score(self, y_true, y_proba):
        
        answer = {}
        for key in self.metrics.keys():
            answer[key] = self.metrics[key](y_true, y_proba)

        return answer
