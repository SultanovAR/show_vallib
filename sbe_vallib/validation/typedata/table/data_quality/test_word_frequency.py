import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from scipy.stats import chi2
import sys, os

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path=os.path.join(path, 'utility')
sys.path.append(path)
from utility import validation_result, make_test_report

#color_text = {'Green':Fore.GREEN, "Yellow":Fore.YELLOW, 'Red':Fore.RED}

def word_frequency_test(
    train,
    oos,
    feature_names = None,
    target_name = 'target',
    p_value = 0.05,
    train_label = 'Train',
    test_label ='Out-of-time',
    key_class = 1,
    **params):
    #init params part
    p_value = params.get('p_value', p_value)
    train_label=params.get('train_label', train_label)
    test_label=params.get('test_label', test_label)
    key_class =params.get('key_class', key_class)
    target_name =params.get('target_name', target_name)
    feature_names = params.get('feature_names', feature_names)
    feature_names = feature_names if feature_names is not None else train.columns

    result = pd.DataFrame(columns=['Наименование класса', f'Кол-во наблюдений {train_label}',
                                   f'Кол-во наблюдений {test_label}', f'Доля {train_label}, %',
                                   f'Доля {test_label}, %', 'p-value', 'Светофор'])

    X_train = train[feature_names]
    X_test = oos[feature_names]
    y_train = train[target_name]
    y_test = oos[target_name]

    len_train = X_train.shape[0]
    len_test = X_test.shape[0]

    for class_name in tqdm(sorted(set(y_train)), leave = True, desc = 'word_frequency_test:'):
        X_train_class = X_train[y_train == class_name]
        X_test_class = X_test[y_test == class_name]
        len_train_class = X_train_class.shape[0]
        len_test_class = X_test_class.shape[0]
        if len_test_class > 0:
            train_freq = np.sum(X_train_class > 0)
            test_freq = np.sum(X_test_class > 0)
            f1 = train_freq[train_freq != 0]
            f2 = test_freq[train_freq != 0]

            n1 = len_train_class
            n2 = len_test_class
            if n1 == 0 or n2 == 0:
                raise ValueError('X_train or X_test has zero length!')

            chi2_obs = np.sum((np.sqrt(n2 / n1) * f1 - np.sqrt(n1 / n2) * f2) ** 2 / (f1 + f2))
            p_value_obs = 1 - chi2.cdf(chi2_obs, n1 - 1)
            f1.sort_values(ascending=False, inplace=True)

            result_p_value = p_value_obs
            color = validation_result(result_p_value, (p_value, -0.01))
        else:
            result_p_value = None
            color = None
        
        result.loc[len(result)] = [class_name, len_train_class, len_test_class,
                                   len_train_class / len_train, len_test_class / len_test,
                                   result_p_value, color]
        
    #result_test = {'table':result, 'color':result['Светофор'][key_class], 'picture':None}
    result_test = make_test_report(table=result, color=result['Светофор'][key_class], title='word_frequency_test' )

    return result_test