from multiprocessing import Semaphore
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import openpyxl
from openpyxl.styles import PatternFill, Font
import itertools
import xlsxwriter
import PIL.Image
import math
import os
from colorama import Fore, Style
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg


def worst_semaphore():
    pass


def load_light(colour):
    dir = os.path.dirname(__file__)
    colour = colour.lower()
    return mpimg.imread(os.path.join(dir, "light_%s.png" % colour))


def table_style(color_list):
    txt_clr = (
        lambda val: "#9c0031"
        if val == "Red"
        else "#9c5700"
        if val == "Yellow"
        else "#006100"
        if val == "Green"
        else None
    )
    bg_clr = (
        lambda val: "#ffc7ce"
        if val == "Red"
        else "#ffeb9c"
        if val == "Yellow"
        else "#c6efce"
        if val == "Green"
        else None
    )
    style = [
        {"index": num, "color": txt_clr(val), "backgroundColor": bg_clr(val)}
        for num, val in enumerate(color_list)
    ]
    return style


def make_test_report(
    color=None,
    table=None,
    title="",
    color_list=[None],
    name=__file__,
    short_res="",
    long_res="",
):
    result_test = {
        "result": {
            "semaphore": {"color": color, "title": title},
            "table": table,
            "tablestyle": table_style(color_list),
        },
        "result_out": pd.DataFrame(
            {
                "name": ["metric_stability_test"],
                "color": [color],
                "short_result": [short_res[:-1]],
                "long_result": [long_res],
                "time": [str(datetime.datetime.today())],
            }
        ),
    }
    return result_test


def worst_color(colors):
    allc = ["Red", "Yellow", "Green"]
    for c in allc:
        if c in set(colors):
            return c
    return None


def semafore_agregation(table_test, type_agg="standart"):
    semaphore = None
    if type_agg == "standart":
        return worst_color(
            list(
                table_test["results"]
                .apply(lambda x: x["result"]["semaphore"]["color"])
                .values
            )
        )  # min по всем тестам

    elif type_agg == "modify_stability_metric":
        stab_metric = table_test[table_test["name_test"] == "metric_stability_test"][
            "results"
        ].values[0]["result"]["semaphore"]["color"]
        # del dict_color_test['metric_stability_test']
        other_metrics = worst_color(
            list(
                table_test[table_test["name_test"] != "metric_stability_test"][
                    "results"
                ]
                .apply(lambda x: x["result"]["semaphore"]["color"])
                .values
            )
        )
        if other_metrics in ["Green", "Yellow"]:
            return stab_metric
        else:
            if stab_metric == "Green":
                return "Yellow"
            else:
                return "Red"

    elif type_agg == "modify_quality_data":
        block_res = table_test.groupby(by="block")["results"].agg(
            lambda x: worst_color(
                list(map(lambda y: y["result"]["semaphore"]["color"], x))
            )
        )
        block_res.reset_index(inplace=True)
        block_res.columns = ["block", "color"]
        quality_metrics = list(
            block_res[block_res["block"] == "quality_data"]["color"].values
        )[0]
        other_metrics = worst_color(
            list(block_res[block_res["block"] != "quality_data"]["color"].values)
        )
        if other_metrics in ["Green", "Yellow"]:
            return quality_metrics
        else:
            if quality_metrics == "Green":
                return "Yellow"
            else:
                return "Red"
    else:
        pass
    return semaphore


def numpy_nan_to_none(val):
    try:
        if np.isnan(val):
            return None
    except TypeError:
        pass
    return val

def parse_params(args):
    if numpy_nan_to_none(args) is None:
        return {}
    params = [i.strip().split("=") for i in args.split("\n")]
    params = {i[0].strip(): eval(i[1].strip()) for i in params}
    return params


def validation_result_reversed(value, boundaries):
    """
    Determine colour of test from value and boundaries. Lesser value means better result.

    :param value: test value. Format: float
    :param boundaries: thresholds for test. Format: (yellow_value, red_value)
    :return: string colour: 'Green', 'Yellow' or 'Red'.
    """
    if value <= min(boundaries):
        return "Green"
    elif value > max(boundaries):
        return "Red"
    else:
        return "Yellow"
    
    
def validation_result(value, boundaries):
    """
    Determine colour of test from value and boundaries. Higher value means better result.

    :param value: test value. Format: float
    :param boundaries: thresholds for test. Format: (yellow_value, red_value)
    :return: string colour: 'Green', 'Yellow' or 'Red'.
    """
    if value <= min(boundaries):
        return "Red"
    elif value > max(boundaries):
        return "Green"
    else:
        return "Yellow"
    
    
def pass_none_values(func):
    """Wrap the function to do nothing if the first parameter is None of any kind"""

    def function_wrapper(*args, **kwargs):
        # Since we expect text input at second position (the first is self), we hard code args[1] check

        # Screw numpy NaN...
        if numpy_nan_to_none(args[1]) is None:
            return -1

        return func(*args, **kwargs)

    return function_wrapper