
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
  - [Пример с выводом картинки](#пример-с-выводом-картинки)
  - [Пример работы с precomputed](#пример-работы-с-precomputed)
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

Необходимо отнаследоваться от ```sbe_vallib.sampler.base.BaseSampler``` и переопределить методы на ваш лад (не забудьте про self.source_state). В качестве примера, можно посмотреть на ```sbe_vallib.sampler.supervised_sampler.SupervisedSampler```, это sampler, подходящий Sampler, когда обучающая выборка это строки вида ```(x, y)```.

### Table 

Пока пусто

### NLP 

Считаем, что вход модели это текст. Значит Sampler должен выдавать список текстов.

### CV 

Считаем, что Sampler хранить пути до картинок, которые будут подаваться на вход модели.

### TimeSeries 

Пока пусто

## Как написать scorer 

Необходимо отнаследоваться от ```sbe_vallib.scorer.base.BaseScorer``` и переопределить метод ```calc_metrics```, в который будут передаваться model, sampler и что-то еще (зависит от тестов). Для примера можно посмотреть на реализацию ```sbe_vallib.scorer.table_scorer.BinaryScorer```.

Выход ```cal_metrics``` должен иметь следующий вид:

```python
  {'metric_name_1': value_1, 'metric_name_2': value_2}
```

или в случае мультиклассовой классификации (table, nlp, cv - не важно)

```python
{'metric_name_1':
  {'class_1': value_11,
   'class_2': value_12,
   'micro': value_micro,
   'macro': value_macro},
 'metric_name_2':
  {'class_1': value_21,
   'class_2': value_22,
   'micro': value_micro,
   'macro': value_macro},}
```

Если в тесте нужно знать размеры классов, то требуем от выхода 'scorer.calc_metrics' дополнительную метрику 'support' то есть в словаре будет запись

```python
{'support':
  {'class_1': support_class_1,
   'class_2': support_class_2}
}
```

### Table 

В table тестах в ```calc_metrics``` также передается флаг ```use_preds_from_sampler``` на его основе решается, брать ли предикты из Sampler по ключу ```'y_pred'``` или же запустить 'честный' предикт модели.

### NLP 

Если это Scorer для NER задачи то он должен реализовывать подсчет метрик для способов поиска совпадаения: exact, strict, partial, type. Подробнее об этом можно почитать [тут](https://github.com/MantisAI/nervaluate)

### CV 

Пока пусто

### TimeSeries 

Пока пусто

## Как написать model wrapper

Надо бы завести список примеров для wrapper'ов, DreamML wrapper AutoNER Wrapper и т.д.

### Table 

Все тесты вызывают метод ```predict_proba``` для получения предсказаний

### NLP 

Все тесты вызывают метод ```predict``` для получения предсказаний

### CV 

Пока пусто

### TimeSeries 

Пока пусто

## Как написать test 

Тест это функция. Она реализует подсчет теста и его оформление(сфетофор, таблица и рисунки). На эту функцию накладываются ограничения на сигнатуру: 

- функция как минимум должна принимать параметры ```model, scorer, sampler, precomputed: dict=None, **kwargs``` 
- функция должна выдать результат работы в следующем формате (привет SberDS)
  
    ```python
    {
        "semaphore": str one of {"gray", "green", "yellow", "red"},
        "result_dict": python object,
        "result_dataframes": List[pd.DataFrame],
        "result_plots": List[PIL.Image],
    }
    ```

    где,
    - "semaphore" -- светофор выставленный за тест
    - "result_dict" -- python object, который валидатор посчитает полезным для дальнейшего использования
    - "result_dataframes" -- список таблиц, которые будут отражены в агрегированном excel файле
    - "result_plots" -- список картинок, которые будут отражены в агрегированном excel файле.

- ```sampler``` это единственный источник данных
- ```precomputed``` это Dict, который служит для обмена предпосчитанными значениями между тестами (зачем 5 раз считать метрику на oos, если можно 1 раз посчитать и записать это значение в dict)

Для единообразия предлагаем писать тесты по этому шаблону:

```python
def test_name(model, sampler, scorer, additional_param_1, additional_param_1, precomputed=None, **kwargs):
  #Проверка входных данных
  if not isinstance(additional_param_1, int):
    raise ValueError(f'additional_param_1 should be "int" instead of "{type(additional_param_1)}"')
  
  #обрабатываем precomputed
  if precomputed is not None:
    if 'value' in precomputed:
      value = precomputed['value']
    else:
      value = calc_value(...)
      precomputed['value'] = value

  #делаем какие-то подсчеты
  useful_values = calc_useful_values(value, ...)

  #проставляем светоформ, рисуем картинки и делаем таблицы
  result = report_test_name(useful_values)
  return result
```

### Пример с выводом картинки

Можете посмотреть код ```table.model_quality.test_ci.test_ci```

### Пример работы с precomputed

Можете посмотреть код ```table.model_quality.test_ci.test_key_metric```

## Как написать pipeline 

Пока пусто

### Pipeline в виде excel файла 

Пока пусто

### Pipeline в виде JSON 

Пока пусто

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

