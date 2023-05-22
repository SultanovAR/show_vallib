import os
import itertools
import math
import typing as tp
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import xlsxwriter

from sbe_vallib.parser import get_callable_from_path
from sbe_vallib.utils.report_helper import worst_semaphore
from sbe_vallib.utils.image import PIL2IOBytes


class Aggregator:
    def __init__(self, save_path: tp.Union[str, os.PathLike],
                 default_purpose: str = 'Не указана',
                 default_interpretation: str = 'Не указана',
                 default_thresholds: tp.Tuple = (3 * ['Не указан']),
                 global_width: int = 800,
                 max_col_width: int = 150):
        """
        :param save_path: path where the excel file will be saved
        :param default_purpose: Set the default value of the purpose field for each test
        :param default_interpretation: Set the default interpretation for each test
        :param default_thresholds: Set the default values for the thresholds  tp.Tuple=(red_value, yellow_value, green_value)
        :param global_width: global width of the excel
        :param max_col_width: Set the width of the columns in excel
        """
        self.save_path = save_path
        self.default_purpose = default_purpose
        self.default_interpretation = default_interpretation
        self.default_thresholds = default_thresholds
        self.workbook = XlsxBot(save_path, "summarization",
                                global_width=global_width, max_col_width=max_col_width)

    def agregate(self,
                 test_results: dict,
                 tests_desc: dict,
                 aggregation_mode_by_block: tp.Optional[dict] = None):
        """
        Creates an excel file with aggregated results of tests

        :param test_results: dict with test_results in the following format:
            {
                'test_key': {
                    "semaphore": str one of {"gray", "green", "yellow", "red"},
                    "result_dict": python object,
                    "result_dataframes": List[pd.DataFrame],
                    "result_plots": List[PIL.Image],
                }
            }

        :param tests_desc: description of a test in the following format:
            {
                'test_key': {
                    'block_key': str,
                    'Название': tp.Optional[str],
                    'Цель': tp.Optional[str],
                    'Интерпретация': tp.Optional[str],
                    'Границы красный': tp.Optional[str],
                    'Границы желтый': tp.Optional[str],
                    'Границы зеленый': tp.Optional[str],
                    **some other keys**}
            }

        :param aggregation_mode_by_block: dict with information how to aggregate results which corresponds to the following example:
            {
                'data_quality': {
                    'print_name': 'Качество данных',
                    'func': 'sbe_vallib.utils.report_helper.worst_semaphore'
                }
            }
        """

        if aggregation_mode_by_block:
            self.make_summarization_sheet(
                test_results, tests_desc, aggregation_mode_by_block)

        for test_key in test_results:
            self.workbook.set_active_sheet(test_key)

            title = test_key
            purpose = self.default_purpose
            interpretation = self.default_interpretation
            thresholds = self.default_thresholds

            if test_key in tests_desc:
                title = tests_desc[test_key].get('Название', title)
                purpose = tests_desc[test_key].get('Цель', purpose)
                interpretation = tests_desc[test_key].get(
                    'Интерпретация', interpretation)
                thresholds = []
                for i, color in enumerate(['красный', 'желтый', 'зеленый']):
                    thresholds.append(tests_desc[test_key].get(
                        f'Границы {color}', self.default_thresholds[i]))

            self.workbook.insert_title(title)

            for table in test_results[test_key]['result_dataframes']:
                self.workbook.insert_table(table)

            if not pd.isna(thresholds).all():
                self.workbook.insert_thresholds_info(*thresholds)

            for ind, pil_img in enumerate(test_results[test_key]['result_plots']):
                self.workbook.insert_image(
                    pil_img, image_name=f'{test_key}_{ind}')

            if not pd.isna(purpose):
                self.workbook.insert_titled_text(str(purpose), 'Цель теста:')

            if not pd.isna(interpretation):
                self.workbook.insert_titled_text(
                    str(interpretation), 'Интерпретация')

    def make_summarization_sheet(self, test_results, tests_desc, aggregation_mode_by_block):
        self.workbook.set_active_sheet('summarization')
        self.workbook.insert_title('Суммаризация результатов по блокам')
        df_table = self.aggregate_block_semaphores(
            test_results, tests_desc, aggregation_mode_by_block)
        self.workbook.insert_table(df_table)
        for i in ['none', 'green', 'yellow', 'red']:
            img = Image.open(os.path.join(os.path.dirname(
                __file__), f"images/semaphores/light_{i}.png"))
            self.workbook.insert_image(img, i, shift=np.array([0, 1]))

    def aggregate_block_semaphores(self, test_results, tests_desc, aggregation_mode_by_block):
        semaphores_by_block = defaultdict(list)
        for test_key in test_results:
            block = tests_desc[test_key].get('block_key', 'custom')
            semaphore = test_results[test_key]['semaphore']
            semaphores_by_block[block].append(semaphore)

        table = {}
        for block_key in semaphores_by_block:
            if block_key in aggregation_mode_by_block:
                aggregate_func = get_callable_from_path(
                    aggregation_mode_by_block[block_key]['func']
                )
                block_print_name = aggregation_mode_by_block[block_key]['print_name']
            else:
                aggregate_func = worst_semaphore
                block_print_name = 'Дополнительные'
            table[block_key] = {
                'Название блока': block_print_name,
                'Результат теста': aggregate_func(semaphores_by_block[block_key]),
                'Кол-во тестов': len(semaphores_by_block[block_key])}

        return pd.DataFrame(table).T


class XlsxBot:
    DEFAULT_CELL_WIDTH = 8.43  # 64 pixels - default Excel cell width
    DEFAULT_CELL_PIXELS = 64
    PIXEL_TO_WIDTH_RATIO = DEFAULT_CELL_WIDTH / DEFAULT_CELL_PIXELS

    ROW_HEIGHT = 20
    CHAR_WIDTH_PIXELS = 8 * 1.125
    GLOBAL_WIDTH_PIXELS = 800
    MAX_CELL_WIDTH_PIXELS = 150 * 2

    MAX_CELL_WIDTH = MAX_CELL_WIDTH_PIXELS / DEFAULT_CELL_WIDTH
    MIN_CELL_WIDTH = DEFAULT_CELL_WIDTH

    def __init__(self, excel_path, sheet_name='Sheet1', global_width=800, max_col_width=150):
        self.GLOBAL_WIDTH_PIXELS = global_width
        self.MAX_CELL_WIDTH_PIXELS = max_col_width
        self.MAX_CELL_WIDTH = self.MAX_CELL_WIDTH_PIXELS / self.DEFAULT_CELL_WIDTH

        self.excel_path = excel_path
        self._wb = xlsxwriter.Workbook(excel_path)
        self._wb.nan_inf_to_errors = True
        self._formats = self._generate_formats()
        self._cursor = np.array([0, 0])
        self._sheet = None
        self.set_active_sheet(sheet_name)

    def set_active_sheet(self, name):
        if name not in self._wb.sheetnames:
            self._sheet = self._wb.add_worksheet(name)
            self._sheet.set_column(
                'A:ZZ', self.DEFAULT_CELL_WIDTH, self._formats.get('cell_default'))
        self._sheet = self._wb.sheetnames[name]
        self._cursor = np.array([0, 0])

    def _generate_formats(self):
        num_formats = {
            'percent': {'num_format': '0.00%'},
            'float': {'num_format': '0.000'},
            'int': {'num_format': '0'},
            None: {}
        }
        colors = {
            'green': {'bg_color': '#C6EFCE', 'font_color': '#006100'},
            'yellow': {'bg_color': '#FFEB9C', 'font_color': '#9C5700'},
            'red': {'bg_color': '#FFC7CE', 'font_color': '#9C0006'},
            'gray': {'bg_color': '#B8B2B1', 'font_color': '#424242'},
            'grey': {'bg_color': '#B8B2B1', 'font_color': '#424242'},
            None: {}
        }
        borders = ['bottom', 'right', 'bottom_right', None]

        p = itertools.product(num_formats, colors, borders)
        formats = {}
        for num_format, color, border in p:
            format = self._wb.add_format(
                {**num_formats[num_format], **colors[color]})
            if border and ('right' in border):
                format.set_right()
            if border and ('bottom' in border):
                format.set_bottom()
            formats[(num_format, color, border)] = format

        formats['table_title'] = self._wb.add_format(
            {'bold': True, 'border': True, 'align': 'center', 'valign': 'vcenter', 'text_wrap': 1})
        formats['sheet_header'] = self._wb.add_format(
            {'bold': True, 'font_size': 14})
        formats['local_header'] = self._wb.add_format(
            {'bold': True, 'italic': True})
        formats['textbox'] = self._wb.add_format({})
        formats['red_threshold'] = self._wb.add_format(
            {'bold': True, 'font_color': '#C00000', 'bg_color': 'white'})
        formats['yellow_threshold'] = self._wb.add_format(
            {'bold': True, 'font_color': '#FFC000', 'bg_color': 'white'})
        formats['green_threshold'] = self._wb.add_format(
            {'bold': True, 'font_color': '#006100', 'bg_color': 'white'})
        formats['cell_default'] = self._wb.add_format(
            {'bg_color': 'white'})  # TODO: merge with other formats
        # formats['table_index'] = self.wb.add_format({'bold': True, 'border': True})
        return formats

    def get_col_formats(self, df):
        col_formats = []
        for c in df.keys():
            if isinstance(c, str) and ('%' in c):
                format = 'percent'
            if df[c].dtype == np.float64:
                format = 'float'
            elif df[c].dtype == np.int64:
                format = 'int'
            else:
                format = None
            col_formats.append(format)
        return col_formats

    def str_convert_except_numbers(self, value):
        if isinstance(value, (int, float, complex)):
            return value
        return str(value)

    def printed_value_width(self, value):
        return len(str(value)) * self.CHAR_WIDTH_PIXELS * self.PIXEL_TO_WIDTH_RATIO

    def adjust_col_width(self, col_ind, text):
        text_len = self.printed_value_width(text)
        width = np.clip(text_len, self.MIN_CELL_WIDTH, self.MAX_CELL_WIDTH)
        self._sheet.set_column(col_ind, col_ind, width,
                               self._formats.get('cell_default'))

    def get_col_width(self, col):
        if col in self._sheet.col_info:
            return self._sheet.col_info[col][0]
        return self.DEFAULT_CELL_WIDTH

    def insert_table(self, df):
        """rows will be colored corresponds to 'Результат теста' column or 'semaphore' column"""
        col_formats = self.get_col_formats(df)

        index_len = max(self.printed_value_width(value) for value in df.index)
        cur_width = max(self.get_col_width(self._cursor[1]), index_len)
        self._sheet.set_column(self._cursor[1], self._cursor[1],
                               cur_width,
                               self._formats.get('cell_default'))

        for i, colname in enumerate(df.columns):
            if self.get_col_width(i + self._cursor[1] + 1) == self.DEFAULT_CELL_WIDTH:
                self.adjust_col_width(i + self._cursor[1] + 1, colname)
            self._sheet.write(self._cursor[0], i + self._cursor[1] + 1,
                              colname, self._formats.get('table_title'))

        for i, row_content in enumerate(df.iterrows()):
            index, values = row_content
            row_color = values.get(
                "Результат теста") or values.get('semaphore')
            row_color = row_color.lower() if row_color else row_color

            self._sheet.write(self._cursor[0] + i + 1, self._cursor[1],
                              index, self._formats.get('table_title'))
            for j, val in enumerate(values):
                col_format = col_formats[j]

                bord_format = []
                if i == len(df) - 1:
                    bord_format.append('bottom')
                if j == len(values) - 1:
                    bord_format.append('right')
                bord_format = '_'.join(bord_format) if len(
                    bord_format) > 0 else None

                self._sheet.write(self._cursor[0] + i + 1,
                                  self._cursor[1] + j + 1,
                                  self.str_convert_except_numbers(val),
                                  self._formats.get((col_format, row_color, bord_format)))

        self._cursor += np.array([len(df) + 2, 0])

    def insert_image(self, pil_image, image_name, shift=None):
        w, h = pil_image.size
        scaling = 1
        if w > self.GLOBAL_WIDTH_PIXELS:
            scaling = self.GLOBAL_WIDTH_PIXELS / w

        self._sheet.insert_image(self._cursor[0], self._cursor[1] + 1, image_name,
                                 {'image_data': PIL2IOBytes(pil_image),
                                  'x_scale': scaling,
                                  'y_scale': scaling})
        if shift is None:
            shift = np.array([math.ceil(h * scaling / self.ROW_HEIGHT) + 1, 0])
        self._cursor += shift

    def insert_text(self, string, format=None, shift=np.array([2, 0])):
        self._sheet.write(self._cursor[0], self._cursor[1] + 1, string, format)
        self._cursor += shift

    def insert_title(self, title):
        self.insert_text(title, self._formats.get('sheet_header'))

    def insert_titled_text(self, text, title):
        text_height = math.ceil(
            len(text) * self.CHAR_WIDTH_PIXELS / self.GLOBAL_WIDTH_PIXELS) + 1
        self._sheet.insert_textbox(self._cursor[0] + 1, self._cursor[1] + 1, text,
                                   {'width': self.GLOBAL_WIDTH_PIXELS,
                                    'height': text_height * self.ROW_HEIGHT,
                                    'object_position': 1})
        self.insert_text(title, self._formats.get(
            'local_header'), shift=0)
        self._cursor += np.array([text_height + 2, 0])

    def insert_thresholds_info(self, red, yellow, green):
        shift = np.array([1, 0])
        self.insert_text("Границы светофора:",
                         self._formats.get('local_header'),
                         shift=shift)
        self.insert_text(red, self._formats.get('red_threshold'),
                         shift=shift)
        self.insert_text(yellow, self._formats.get('yellow_threshold'),
                         shift=shift)
        self.insert_text(green, self._formats.get('green_threshold'),
                         shift=shift)
        self._cursor += shift

    def __del__(self):
        self._wb.close()
