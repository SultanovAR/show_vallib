import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
'''
import sys, os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
path=os.path.join(path, 'validation')
sys.path.append(path)
#print(sys.path)'''
from validation import TableValidation, utility

X_train = pd.DataFrame(columns = ['1', '2', '3'])
X_oos = pd.DataFrame(columns = ['1', '2', '3'])
X_oot = pd.DataFrame(columns = ['1', '2', '3'])
for frame in [X_train, X_oos, X_oot]:
    for c in ['1', '2', '3']:
        frame[c] = np.random.random(size = 1000)
y_train = pd.Series(np.random.choice([0, 1],size = 1000), name = 'target')
y_oos =  pd.Series(np.random.choice([0, 1], size = 1000), name = 'target')
y_oot =  pd.Series(np.random.choice([0, 1], size = 1000), name = 'target')
lr = LogisticRegression().fit(X_train, y_train)

feats = X_train.columns
train = pd.concat([X_train, y_train], axis =1)
oos = pd.concat([X_oos, y_oos], axis =1)
oot = pd.concat([X_oot, y_oot], axis =1)


#использование собственной метрики качества
def custom_key_metric(y_test, y_predict_proba, **kwargs):
    return 1

# добавление своего теста
def some_new_test( **kwargs):
    result = utility.make_test_report(color='Green', table=pd.DataFrame({'col1':[1]}))
    return result

# замена стандаратного теста
def extremal_missing_values_test(**kwargs):
    result = utility.make_test_report(color='Green', table=pd.DataFrame({'col1':[1]}))
    return result


custom_tests = {
'some_new_test':{'Вызов':some_new_test,'Информативный тест':1,'Длинный тест':0, 'Блок':'New_block' },
'extremal_missing_values_test':{'Вызов': extremal_missing_values_test,'Информативный тест':1,'Длинный тест':1, 'Блок':'data_quality'}
}

validation = TableValidation(train = train, 
                            oos =oos, 
                            oot =oot, 
                            feature_names=feats,
                            model = lr,
                            pipeline = '31', 
                            verbose =[],
                            exclude_list = ['train_test_independence_test', 'test_1_8_oot',
                             'test_4_2','test_5_4','test_2_2','test_3_1','test_5_7','test_5_6',
                             'test_2_5','test_4_1','test_3_2'], #исключение стандартных тестов
                            **{#'straregy_bussines' : 'score',
                               'key_metric' : custom_key_metric, 
                               'custom_tests':custom_tests
                               }
                           )
validation.validate()