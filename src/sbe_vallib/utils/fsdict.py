import os
from collections.abc import MutableMapping

import joblib


class FSDict(MutableMapping):
    """
    Persistent dict is based on FileSystem and joblib.
    It works faster with numpy and pandas objects.
    The interface is the same as Dict, except for the __init__ method.
    """

    def __init__(self, store_dir_path: str,
                 compress: int = 0):
        """
        :param store_dir_path: str: Specify the directory to store the data in
        :param compress: int: Specify the compression level for the file from 0 to 9
        """
        self.store_dir_path = store_dir_path
        self.compress = compress
        os.makedirs(store_dir_path, exist_ok=True)

    def __setitem__(self, key, value):
        file_path = os.path.join(self.store_dir_path, key)
        joblib.dump(value, file_path, compress=self.compress)

    def __getitem__(self, key):
        file_path = os.path.join(self.store_dir_path, key)
        try:
            return joblib.load(file_path)
        except FileNotFoundError:
            raise KeyError(key)

    def __delitem__(self, key):
        file_path = os.path.join(self.store_dir_path, key)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            raise KeyError(key)

    def __len__(self):
        return len(os.listdir(self.store_dir_path))

    def __iter__(self):
        return iter(os.listdir(self.store_dir_path))

    def __repr__(self):
        return f"FileDict keys: {tuple(self)}"
