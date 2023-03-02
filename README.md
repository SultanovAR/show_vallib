
- [Установка](#установка)
- [Введение](#введение)
- [Как запустить](#как-запустить)
- [Как написать sampler](#как-написать-sampler)
  - [Table](#table)
  - [NLP](#nlp)
  - [CV](#cv)
  - [TimeSeries](#timeseries)
- [Как написать scorer](#как-написать-scorer)
  - [Table](#table-1)
  - [NLP](#nlp-1)
  - [CV](#cv-1)
  - [TimeSeries](#timeseries-1)
- [Как написать model wrapper](#как-написать-model-wrapper)
  - [Table](#table-2)
  - [NLP](#nlp-2)
  - [CV](#cv-2)
  - [TimeSeries](#timeseries-2)
- [Как написать test](#как-написать-test)
- [Как написать pipeline](#как-написать-pipeline)
  - [Pipeline в виде excel файла](#pipeline-в-виде-excel-файла)
  - [Pipeline в виде JSON](#pipeline-в-виде-json)
- [Поясняем за utils](#поясняем-за-utils)
  - [fsdict.py](#fsdictpy)
  - [image.py](#imagepy)
  - [metrics.py](#metricspy)
  - [pd\_np\_interface.py](#pd_np_interfacepy)
  - [quantization.py](#quantizationpy)
  - [report\_helper.py](#report_helperpy)



## Установка 

    Там где находится setup.py (там же README.md) выполнить команду ```pip install -e .``` это установит библиотеку в editable mode

## Введение 

Этот readme поможет сделать первые шаги в освоении библиотеки, но конечно все описать невозможно и просто прочтение кода ничего не заменит (мы старались сделать его удобночитаемым).

Библиотека базируется на 6 абстракциях: ```Sampler, Scorer, Model, Test, Pipeline, Validation```.

    Sampler - обертка над данными, также он берет на себя функцию ресамплинга данных(bootstraping, time-backtracking)

    Model - очевидно, что это. Обертки иногда надо будет писать

    Scorer - объект, который считает метрики, если в него подать Sampler(данные) и Model

    Test - функция которая принимает model, scorer, sampler и выдает результат в определенном формате

    Pipeline - отображение методики в нашей бибилиотеке: содержит список тестов, параметры, интерпретации и другое. Это может быть записано в excel формате или же в JSON.

    Validation - запускает все тесты из Pipeline затем формирует отчет

Подробнее что делает каждая абстракция и с какой целью можно увидеть в презентации ```./presentation/presentation.ipynb```


## Как запустить 

- Выбрать Sampler тут ```sbe_vallib.sampler```, если там вам ничего не подходит надо написать самому
- Выбрать Scorer тут ```sbe_vallib.scorer```, если там вам ничего не подходит надо написать самому
- В секции 'как написать Model Wrapper' посмотреть какой интерфейс должен быть в модели (У table, cv, nlp моделей они разные). Написать Wrapper, если у вашей модели другой интерфейс.
- Выбрать Pipeline из готовых ```sbe_vallib.table.pipelines```, ```sbe_vallib.cv.pipelines```, ```sbe_vallib.nlp.pipelines```. Или же написать свой в excel или JSON формате.
- Закинуть это все в Validation и получить готовую валидацию :)

Примеры использования библиотеки для самых различных случаев есть в ```./examples/```

## Как написать sampler 

### Table 

### NLP 

### CV 

### TimeSeries 

## Как написать scorer 

### Table 

### NLP 

### CV 

### TimeSeries 

## Как написать model wrapper 

### Table 

### NLP 

### CV 

### TimeSeries 

## Как написать test 

## Как написать pipeline 

### Pipeline в виде excel файла 

### Pipeline в виде JSON 

## Поясняем за utils 

### fsdict.py 

    Имплементация dict, которая хранит все в файловой системе, а не в оперативке. В библиотеке используется для передачи объектов между тестами (precomputed).

### image.py 

    Помощь в работе с изображениями. Например перевод из matplotlib figure в PIL.image.

### metrics.py 

    Словари с типичными метриками для разных задач.

### pd_np_interface.py 

    За частую у нас вход идет либо np.array, либо pd.DataFrame, либо python объекты.
    У каждого из них свои методы для concat, взятия i-го столбца и тд. Тут собраны функции, которые создают единый интерфейс для работы с этими объектами. Реализованы concat, get_index, get_columns, set_column, all_columns и быть может что-то еще.

### quantization.py

    Класс для квантизации/бининга (непрерывный признак разбить на бины)
### report_helper.py 

    Вспомогательные функции для выставления светофоров и формирования выходного словаря у теста.

