from abc import ABC, abstractmethod
import typing as tp


class BaseSampler(ABC):
    def __init__(
        self,
        train: dict,
        oos: tp.Optional[dict],
        oot: tp.Optional[dict],
        **kwargs
    ):
        """
        Parameters
        ----------
        train : dict
            {"X": Iterable, 'y_true': Iterable, 'y_pred': Optional[Iterable]}
        oos : tp.Optional[dict]
            {"X": Iterable, 'y_true': Iterable, 'y_pred': Optional[Iterable]}
        oot : tp.Optional[dict]
            {"X": Iterable, 'y_true': Iterable, 'y_pred': Optional[Iterable]}
        """
        self.source_train = train
        self.source_oos = oos
        self.source_oot = oot
        self.source_state = True

    def reset(self):
        self.source_state = True

    @property
    def train(self):
        return self.source_train

    @property
    def oos(self):
        return self.source_oos

    @property
    def oot(self):
        return self.source_oot

    @abstractmethod
    def set_state(self):
        self.source_state = False
        pass
