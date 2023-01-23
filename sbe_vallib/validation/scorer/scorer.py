from sbe_vallib.validation.scorer.base import BaseScorer


class Scorer(BaseScorer):
    def __init__(self, metrics, **params):

        super().__init__(metrics, **params)

    def __getitem__(self, key):
        return self.metrics[key]

    def keys(self):
        return self.metrics.keys()

    def score(self, y_target, y_pred):
        """
        Расчет указанных метрик

        @y_target: значения целевой переменной
        @y_pred: предсказания модели
        """

        answer = {}
        if self.calc_every_class:
            """
            Проводим расчет метрики для каждого класса
            """
            pass
        else:
            for key in self.metrics.keys():
                answer[key] = self.metrics[key](y_target, y_pred)

        return answer
