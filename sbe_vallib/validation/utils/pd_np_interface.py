import typing as tp
import pandas as pd
import numpy as np


def is_pandas(data: tp.Union[np.ndarray, pd.DataFrame, pd.Series]):
    """A function to deal with pd.DataFrame, pd.Series and np.array in the same manner"""
    return hasattr(data, 'iloc')


def concat(arrays: tp.List[tp.Union[np.ndarray, pd.DataFrame, pd.Series]]):
    """A function to deal with pd.DataFrame, pd.Series and np.array in the same manner"""
    assert len(arrays) > 0, 'argument list is empty'
    if all([is_pandas(i) for i in arrays]):
        return pd.concat(arrays)
    elif all(isinstance(i, list) for i in arrays):
        return sum(arrays, [])
    return np.concatenate(arrays)


def get_index(data: tp.Union[np.ndarray, pd.DataFrame, pd.Series], index):
    """A function to deal with pd.DataFrame and np.array in the same manner"""
    if is_pandas(data):
        return data.iloc[index]
    elif isinstance(data, list):
        return [data[i] for i in index]
    return data[index]


def get_columns(data: tp.Union[np.ndarray, pd.DataFrame], columns):
    """
    The get_columns function returns a subset of the columns from the input data.
    
    :param data: tp.Union[np.ndarray]: Make the function compatible with both numpy arrays and pandas dataframes
    :param [pd.DataFrame]: Check if the data is a pandas dataframe
    :param columns: Select a subset of columns from the data
    :return: A subset of the columns in data
    :doc-author: Trelent
    """
    if is_pandas(data):
        return data[columns]
    return data[:, columns]


def set_column(where: tp.Union[np.ndarray, pd.DataFrame],
               data: tp.Union[np.ndarray, pd.Series],
               column):
    if is_pandas(where):
        where.loc[:, column] = data
    else:
        where[:, column] = data


def all_columns(data: tp.Union[np.ndarray, pd.DataFrame]):
    if is_pandas(data):
        return data.columns
    if len(data.shape) > 1:
        return np.arange(data.shape[1]).astype(int)
    return None
