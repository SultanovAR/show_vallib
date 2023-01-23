from matplotlib.pyplot import table, title
import pandas as pd
import sys, os

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path=os.path.join(path, 'utility')
sys.path.append(path)
from utility import make_test_report


def train_test_independence_test(train, 
                                oos, 
                                feature_names = None,
                                quantile_yellow = 0.95,
                                quantile_red = 0.99,
                                **params):
    quantile_yellow = params.get('quantile_yellow', quantile_yellow)
    quantile_red = params.get('quantile_red', quantile_red)
    feature_names = params.get('feature_names', feature_names)
    feature_names = feature_names if feature_names is not None else train.columns

    params = train_test_independence_calculate(train,oos,feature_names,**params)

    gini_fact = params['logs']['split_stats']['gini_fact']
    gini_random_list = sorted(params['logs']['split_stats']['gini_random_list'])
    gini_quantile = min(sum(gini_random_list <= gini_fact) / (len(gini_random_list) - 1), 1)

    if (1-quantile_yellow)/2 < gini_quantile < (1+quantile_yellow)/2:
        result_color = "Green"
    elif (1-quantile_red)/2 <= gini_quantile <= (1+quantile_red)/2:
        result_color = "Yellow"
    else:
        result_color = "Red"

    result = pd.DataFrame({'Фактический gini': gini_fact,
                           'Квантиль': gini_quantile,
                           'Результат теста': result_color
                           }, index=[0])

    result_test = make_test_report(table=result, color=result_color, title = 'train_test_independence_test' )
    return result_test

def train_test_independence_calculate(train, oos,feature_names, **params):
    pass





