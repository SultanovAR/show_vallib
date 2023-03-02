# Содержание

1. [Установка](#installation)
2. [Введение](#introduction)
3. [Как запустить](#run)
4. [Как написать sampler](#writesampler)
    1. [Table](#writesampler_table)
    2. [NLP](#writesampler_nlp)
    3. [CV](#writesampler_cv)
    4. [TimeSeries](#writesampler_ts)
5. [Как написать scorer](#writescorer)
    1. [Table](#writescorer_table)
    2. [NLP](#writescorer_nlp)
    3. [CV](#writescorer_cv)
    4. [TimeSeries](#writescorer_ts)
6. [Как написать model wrapper](#writewrapper)
    1. [Table](#writewrapper_table)
    2. [NLP](#writewrapper_nlp)
    3. [CV](#writewrapper_cv)
    4. [TimeSeries](#writewrapper_ts)
7. [Как написать тест](#writetest)
8. [Как написать pipeline](#writepipeline)
    1. [Pipeline в виде excel файла](#writepipeline_excel)
    2. [Pipeline в виде JSON](#writepipeline_json)
5. [Поясняем за utils](#utils)
    1. [fsdict.py](#fsdict)
    2. [image.py](#image)
    3. [metrics.py](#metrics)
    4. [pd_np_interface.py](#pd_np_interface)
    5. [quantization.py](#quantization)
    6. [report_helper.py](#report_helper)

## Установка <a name="installation"></a>

Там где находится setup.py (там же README.md) выполнить команду ```pip install -e .``` это установит библиотеку в editable mode

## Введение <a name="introduction"></a>

## Как запустить <a name="run"></a>

## Как написать sampler <a name="writesampler"></a>

### Table <a name="writesampler_table"></a>

### NLP <a name="writesampler_nlp"></a>

### CV <a name="writesampler_cv"></a>

### TimeSeries <a name="writesampler_ts"></a>

## Как написать scorer <a name="writescorer"></a>

### Table <a name="writescorer_table"></a>

### NLP <a name="writescorer_nlp"></a>

### CV <a name="writescorer_cv"></a>

### TimeSeries <a name="writescorer_ts"></a>

## Как написать model wrapper <a name="writewrapper"></a>

### Table <a name="writewrapper_table"></a>

### NLP <a name="writewrapper_nlp"></a>

### CV <a name="writewrapper_cv"></a>

### TimeSeries <a name="writewrapper_ts"></a>

## Как написать test <a name="writetest"></a>

## Как написать pipeline <a name="writepipeline"></a>

### Pipeline в виде excel файла <a name="writepipeline_excel"></a>

### Pipeline в виде JSON <a name="writepipeline_json"></a>

## Поясняем за utils <a name="utils"></a>

### fsdict.py <a name="fsdict"></a>

    Имплементация dict, которая хранит все в файловой системе, а не в оперативке. В библиотеке используется для передачи объектов между тестами (precomputed).

### image.py <a name="image"></a>

    Помощь в работе с изображениями. Например перевод из matplotlib figure в PIL.image.

### metrics.py <a name="metrics"></a>

    Словари с типичными метриками для разных задач.

### pd_np_interface.py <a name="pd_np_interface"></a>

    За частую у нас вход идет либо np.array, либо pd.DataFrame, либо python объекты.
    У каждого из них свои методы для concat, взятия i-го столбца и тд. Тут собраны функции, которые создают единый интерфейс для работы с этими объектами. Реализованы concat, get_index, get_columns, set_column, all_columns и быть может что-то еще.

### quantization.py <a name="quantization"></a>

    Класс для квантизации/бининга (непрерывный признак разбить на бины)
### report_helper.py <a name="report_helper"></a>

    Вспомогательные функции для выставления светофоров и формирования выходного словаря у теста.

