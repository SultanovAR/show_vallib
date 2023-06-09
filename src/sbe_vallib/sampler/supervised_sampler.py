import numpy as np

from sklearn.model_selection import train_test_split
from sbe_vallib.sampler.base import BaseSampler

from sbe_vallib.utils.pd_np_interface import concat, get_index


class SupervisedSampler(BaseSampler):
    def __init__(
        self,
        train: dict,
        oos: dict,
        oot: dict = None,
        **kwargs
    ):
        """
        :param train: dict: Store the training data
        :param oos: dict: Store the oos data
        :param oot: dict: Specify a test set that is not part of the training set
        :param bootstrap: bool: Indicate whether the model is trained on bootstrapped data
        :param **kwargs: Pass a variable number of keyword arguments to a function
        :return: None
        """
        super().__init__(train, oos, oot, **kwargs)
        self._seed = None
        self._gen_method = None
        self._stratify = None
        self.index = dict()

    def set_state(self, seed: int, gen_method: str = 'resampling', stratify: bool = False):
        """
        The set_seed function is used to set the seed for the random number generator.
        The purpose of this function is to ensure that we can reproduce our results by
        ensuring that we always use the same seed and permutation when bootstrapping.

        :param seed: int: Set the seed for the random number generator
        :return: The value of the seed
        """
        self.source_state = False
        self._seed = seed
        self._gen_method = gen_method
        self._stratify = stratify
        generator = np.random.default_rng(seed=seed)
        size_train = len(self.source_train["X"])
        size_oos = len(self.source_oos["X"])

        if gen_method == 'bootstrap':
            self.index = {
                "train": generator.integers(0, size_train, size_train),
                "oos": generator.integers(0, size_oos, size_oos),
            }
        elif gen_method == 'resampling':
            target_for_stratify = None
            if stratify:
                target_for_stratify = concat(
                    (self.source_train["y_true"], self.source_oos['y_true'])
                )

            self.index = {'train': None, 'oos': None}
            self.index['train'], self.index['oos'] = train_test_split(np.arange(size_train + size_oos),
                                                                      test_size=size_oos,
                                                                      random_state=seed,
                                                                      shuffle=True,
                                                                      stratify=target_for_stratify)
        else:
            raise ValueError(f'"gen_method": {gen_method} is not implemented')

    def _get_data_by_gen_method(self, data: dict, data_type: str):
        if self.source_state:
            return data

        result = {}
        if (self._gen_method == 'bootstrap'):
            for key in data:
                result[key] = get_index(
                    data[key], self.index[data_type])
        elif (self._gen_method == 'resampling'):
            for key in self.source_train:
                concated = concat(
                    (self.source_train[key], self.source_oos[key]))
                result[key] = get_index(concated, self.index[data_type])
        return result

    @property
    def train(self):
        return self._get_data_by_gen_method(self.source_train, 'train')

    @property
    def oos(self):
        return self._get_data_by_gen_method(self.source_oos, 'oos')

    @property
    def oot(self):
        return self.source_oot
