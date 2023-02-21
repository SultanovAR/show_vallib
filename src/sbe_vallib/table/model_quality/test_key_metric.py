import pandas as pd
from sbe_vallib.sampler import SupervisedSampler





def key_metric_test(
    y_oos,
    y_oos_answer,
    metric="gini",
    thresholds=(0.2, 0.4),
    use_scores=True,
    semafore_reverser=False,
    key_class=None,
    cutoff=0.5,
    top=None,
    **params
):
    pass


#     y_test = y_oos
#     y_test_answer = y_oos_answer
#     metric = params.get('key_metric', metric)
#     if type(metric)==str:
#         metric_name=metric
#     else:
#         metric_name= 'custom'
#     thresholds = params.get('thresholds_key_metric', thresholds)
#     use_scores = params.get('use_scores', use_scores)
#     semafore_reverser = params.get('semafore_reverser', semafore_reverser)
#     classes = params.get('classes', list(set(y_test)))
#     key_class = params.get('key_class', key_class)
#     cutoff = params.get('cutoff', cutoff)
#     top = params.get('top', top)


#     class_names  = np.array(sorted(classes))

#     metric_regression = {}

#     metrics_all_classses = {
#         "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
#         "f1": lambda y_true, y_pred: f1_score(y_true, y_pred,average='macro'),
#         "precision": lambda y_true, y_pred: precision_score(y_true, y_pred,average='macro'),
#         "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')
#     }

#     metrics_one_classes ={
#         'gini':  lambda y_true, y_pred: 2*roc_auc_score(y_true, y_pred)-1,
#         'roc-auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
#         'pr-auc': lambda y_true, y_pred: average_precision_score(y_true, y_pred)
#     }

#     if metric_name in ["accuracy", "f1", "precision", "recall"]:
#         use_scores=False


#     metric_value_test = []
#     #class_names = class_names if key_class is None else [key_class]
#     if metric_regression.get(metric_name, None) is not None:
#         pass
#     elif metrics_one_classes.get(metric_name, None) is not None:
#         metric = metrics_one_classes[metric_name]
#         for class_ in class_names:
#             if use_scores:
#                 y_test_result_alg = y_test_answer[:, class_]
#             else:
#                 y_test_result_alg = np.array(list(map(lambda x: int(class_names[x.index(max(x))]==class_), y_test_answer)))
#             metric_value_test.append(  metric(y_test, y_test_result_alg))
#     elif metrics_all_classses.get(metric_name, None) is not None:
#         metric = metrics_all_classses[metric_name]
#         if use_scores:
#             #больные ублюдки
#             pass
#         else:
#             if len(class_names)==2:
#                 if key_class is None:
#                     key_class = class_names[1]
#                 no_key_class = list(set(class_names) - set([key_class]))[0]
#                 if top is None:
#                     # вариант, когда отсекаем по cutoff
#                     y_test_result_alg = np.array(list(map(lambda x: key_class if x>cutoff else no_key_class, y_test_answer[:,class_names.index(key_class) ])))
#                 else:
#                     pass# вариант, когда отсекаем по топу
#             else:
#                 y_test_result_alg = np.array(list(map(lambda x: class_names[x.index(max(x))], y_test_answer)))
#         metric_value_test.append(  metric(y_test, y_test_result_alg))

#     else:
#         for class_ in class_names:
#             if use_scores:
#                 y_test_result_alg = y_test_answer[:, class_]
#             else:
#                 y_test_result_alg = np.array(list(map(lambda x: int(class_names[x.index(max(x))]==class_), y_test_answer)))
#             metric_value_test.append(metric(y_test, y_test_result_alg, **params))


#     test_result = []


#     if semafore_reverser:
#         test_result = list(map(lambda metr: validation_result_reversed(metr, thresholds), metric_value_test))
#         test_result_global = validation_result_reversed(np.mean(metric_value_test), thresholds)
#     else:
#         test_result=list(map(lambda metr: validation_result(metr, thresholds), metric_value_test))
#         test_result_global = validation_result(np.mean(metric_value_test), thresholds)
#     if key_class is not None:
#         test_result_global = test_result[class_names.index(key_class)]
#     result = pd.DataFrame({'Наименование класса': class_names,
#                            'Значение ' + metric_name: metric_value_test,
#                            'Результат теста': test_result
#                               })
#     color_list = np.array(test_result)
#     if test_result_global == "Green":
#         result_text = "Metric values are within green threshold levels."
#     elif test_result_global == "Yellow":
#         weak_classes = ', '.join(list(map(str,class_names[np.where(color_list == "Yellow")])))
#         result_text = f"Metric values for {weak_classes} are within yellow threshold levels."
#     elif test_result_global == "Red":
#         weak_classes = ', '.join(list(map(str,class_names[np.where(color_list == "Red")])))
#         result_text = f"Metric values for {weak_classes} exceed critical threshold level."
#     result_test = make_test_report(color=test_result_global,title=result_text,table=result,color_list=color_list )
#     return result_test
