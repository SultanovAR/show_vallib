import pandas as pd
import numpy as np
from collections import Counter

from sbe_vallib.utility import validation_result, get_pred_method
from sbe_vallib.utils.logger import Logger

def test_metric(
    scorer=None,
    sampler=None,
    X_sample=None,
    y_sample=None,
    y_pred=None,
    metric_func=None,
    model=None,
    metric_name="f1",
    average="macro",
    sample="oos",
    class_names=[],
    min_class_elements=20,
    path=".",
    thresholds_metric=(0.4, 0.6),
    test_name="test_key_metric",
    verbose=[
        "test_warnings",
        "test_errors",
        "semaphore",
        "model_errors",
        "model_warning",
        "progress",
        "short_res",
        "pictures",
        "info",
    ],
    logging=[
        "test_warnings",
        "test_errors",
        "semaphore",
        "model_errors",
        "model_warning",
        "progress",
        "short_res",
        "pictures",
        "answers",
        "tests_logs",
        "info",
    ],
    logger=None,
):
    save_excel = True if "tests_logs" in logging else False
    if logger is None:
        logger = Logger(path, level=logging)

    if X_sample is None:
        X_sample, y_sample, y_pred = sampler[sample]
    
    if y_pred is None:
        pred_func = get_pred_method(model)
        y_pred = pred_func(model, X_sample)

    if metric_func is None:
        metric_func = scorer[metric_name]

    # ---
    # добавить assert-ы
    # ---

    if len(y_sample.shape) == 1:
        class_counts = np.array([np.sum(y_sample == class_name) for class_name in class_names])
    else:
        class_counts = np.array([np.sum(y_sample.iloc[:, i]) for i in range(len(class_names))])
    class_counts = np.append(class_counts, len(y_sample))

    labels = class_names if len(y_sample.shape) == 1 else np.arange(len(class_names))
    metric_values = metric_func(y_sample, y_pred, labels=labels, average=None)

    labels = labels[class_counts[:-1] >= min_class_elements]
    metric_values.append(metric_func(y_sample, y_pred, labels=labels, average=average))        

    target_metrics = [f'{class_name}_{metric_name}' for class_name in class_names]
    target_metrics.append(f'{average}-{metric_name}')

    semaphore_colors = []

    for metric, class_count, metric_value in zip(target_metrics, class_counts, metric_values):
        if average not in metric:
            if class_count >= min_class_elements:
                semaphore_colors.append(validation_result(metric_value, thresholds_metric))
            else:
                semaphore_colors.append('Grey')
        else:
            current_color = validation_result(metric_value, thresholds_metric)
            color_counts = Counter(semaphore_colors)
            semaphore_color = get_color_for_macro_metric(current_color, color_counts, len(class_names))
            semaphore_colors.append(semaphore_color)

    class_names = list(class_names) + [average]
    result = pd.DataFrame({'Наименование класса': class_names,
                            'Коэффициент ' + metric_name: metric_values,
                            'Размер класса': class_counts,
                            'Результат теста': semaphore_colors})

    if save_excel:
        logger.save(result, test_name + "_" + metric_name)
    
    test_res = result[result["Наименование класса"] == average]["Результат теста"]
    if "semaphore" in verbose:
        print("Результат теста {} : {}".format(test_name, test_res))

    if "short_res" in verbose:
        print("Тест {}".format(test_name))
        print(result.to_dict())

    logger("semaphore", "Результат теста {}: {}".format(test_name, test_res))
    logger("short_res", "Тест {}".format(test_name))
    logger("short_res", result.to_dict())

    return result


def get_color_for_macro_metric(current_color, color_counts, class_count):
    COLOR_COUNTS_THRESHOLD = 0.2

    if current_color == 'Red' or color_counts.get('Red', 0) / class_count > COLOR_COUNTS_THRESHOLD:
        return 'Red'
    elif current_color == 'Yellow' or color_counts.get('Yellow', 0) / class_count > COLOR_COUNTS_THRESHOLD \
            or color_counts.get('Red', 0):
        return 'Yellow'
    else:
        return 'Green'