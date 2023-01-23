import sys, os

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path=os.path.join(path, 'utility')
sys.path.append(path)
from utility import validation_result_reversed,validation_result, make_test_report, agg_two_criterion
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
path=os.path.join(path, 'model_quality', 'test_key_metric')
sys.path.append(path)
from model_quality.test_key_metric import key_metric_test


def metric_stability_test(
    y_oos, 
    y_oos_answer,
    y_train=None, 
    y_train_answer=None,
    y_oot=None,
    y_oot_answer=None,
    metric = 'gini',
    thresholds_abs=(0.1, 0.2), 
    thresholds_rel= (0.15, 0.25),
    thresholds_key_metric = (0.2, 0.4),
    use_scores=True,
    key_class = None, 
    cutoff= 0.5,
    top= None,
    semafore_reverser=False, 
    **params):

    metric = params.get('key_metric', metric)
    if type(metric)==str:
        metric_name=metric
    else:
        metric_name= 'custom'
    thresholds_abs = params.get('thresholds_abs', thresholds_abs)
    thresholds_rel = params.get('thresholds_rel', thresholds_rel)
    thresholds_key_metric = params.get('thresholds_key_metric', thresholds_key_metric)
    use_scores = params.get('use_scores', use_scores)
    key_class = params.get('key_class', key_class)
    cutoff = params.get('cutoff', cutoff)
    top = params.get('top', top)
    semafore_reverser = params.get('semafore_reverser', semafore_reverser)

    #проверка на лишние None
    if ((y_train is not None) and (y_train_answer is not None)) or ((y_oot is not None) and (y_oot_answer is not None)):
        pass
    else:
        print('error stability test')
        
    test_result_global = 'grey'
    if y_oot is None:
        new_params = copy.deepcopy(params)
        baseline_params = copy.deepcopy(params)
        baseline_params['oos'] = new_params['train'] 
        label_baseline = 'train'
        label_test = 'out-of-sample'
        baseline_test= y_train 
        baseline_answer=y_train_answer
        new_test=y_oos
        new_answer=y_oos_answer
    else:
        new_params = copy.deepcopy(params)
        baseline_params = copy.deepcopy(params)  
        new_params['oos'] = new_params['oot'] 
        label_baseline = 'out-of-sample'
        label_test = 'out-of-time'
        baseline_test= y_oos #params.get('oos', pd.Series(y_oos, name='target').to_frame())['target']
        baseline_answer=y_oos_answer
        new_test=y_oot
        new_answer=y_oot_answer
        
    base_name_columns = ['Наименование класса', 'Значение ' + metric_name]

    result_baseline = key_metric_test(  baseline_test, 
                                        baseline_answer,
                                        metric=metric, 
                                        thresholds=thresholds_key_metric, 
                                        use_scores=use_scores,
                                        semafore_reverser=semafore_reverser,
                                        key_class=key_class,
                                        cutoff=cutoff,
                                        top=top,
                                        **baseline_params)["result"]['table'][base_name_columns]  
    result_baseline.columns = ['Наименование класса', 'Значение ' + metric_name + ' на выборке ' + label_baseline]
    result_test = key_metric_test(      new_test, 
                                        new_answer,
                                        metric=metric, 
                                        thresholds=thresholds_key_metric, 
                                        use_scores=use_scores,
                                        semafore_reverser=semafore_reverser,
                                        key_class=key_class,
                                        cutoff=cutoff,
                                        top=top,
                                        **new_params)["result"]['table'][base_name_columns]  
    result_test.columns = ['Наименование класса', 'Значение ' + metric_name + ' на выборке ' + label_test]
    
    result = result_baseline.merge(result_test, on ='Наименование класса', how = 'outer')
    result['Падение ' + metric_name + ' абс-ое'] = result['Значение ' + metric_name + ' на выборке ' + label_baseline] - result['Значение ' + metric_name + ' на выборке ' + label_test]
    result['Падение ' + metric_name + ' отн-ое, %'] = result.apply(lambda x: x['Падение ' + metric_name + ' абс-ое']/x['Значение ' + metric_name + ' на выборке ' + label_baseline], axis = 1 )
    result.fillna(np.nan, inplace = True)
    class_names = result['Наименование класса'].values

    abs_light =  validation_result_reversed(np.mean(result['Падение ' + metric_name + ' абс-ое']), thresholds_abs)
    rel_light = validation_result_reversed(np.mean(result['Падение ' + metric_name + ' отн-ое, %']), thresholds_rel)

    if y_oot is not None:
        test_result_global = agg_two_criterion(abs_light, rel_light)
        
    if key_class is not None:
        test_result = [test_result_global]
        class_names = [key_class]
    else:
        test_result= []
        for i in range(len(class_names)):
            abs_light =  validation_result_reversed( result['Падение ' + metric_name + ' абс-ое'][i], thresholds_abs)
            rel_light = validation_result_reversed(result['Падение ' + metric_name + ' отн-ое, %'][i], thresholds_rel)
            test_result.append(agg_two_criterion(abs_light, rel_light))

    result['Результат теста'] = test_result
    #print(test_result_global)
   
        
        
    #result_test = {'table':result, 'color':test_result_global, 'picture':None}# мой вывод
    
    #вывод сбер.дс
    color_list = test_result
    if test_result_global == "Green":
        result_text = "Metric values are within green threshold levels."
    elif test_result_global == "Yellow":
        weak_classes = ', '.join(list(map(str,class_names[np.where(color_list == "Yellow")])))
        result_text = f"Metric values for {weak_classes} are within yellow threshold levels."
    elif test_result_global == "Red":
        weak_classes = ', '.join(list(map(str,class_names[np.where(color_list == "Red")])))
        result_text = f"Metric values for {weak_classes} exceed critical threshold level."

    result_test=make_test_report(color=test_result_global, table = result,
            title=result_text,    color_list=color_list,    name='metric_stability_test ' + metric_name , )

    return result_test