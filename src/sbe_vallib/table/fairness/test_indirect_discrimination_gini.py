
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from sbe_vallib.utils.cat_features import get_cat_features
from sbe_vallib.utils.quantization import Quantizer
import sbe_vallib.utils.pd_np_interface as interface
from sbe_vallib.utils.metrics import gini_score


def leave_presented_classes(y_pred, classes, presented_classes):
    class_to_index = {cl: i for i, cl in enumerate(classes)}
    presented_indices = [class_to_index[i] for i in presented_classes]
    y_pred = np.array(y_pred)
    y_pred = y_pred[:, presented_indices]
    y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)
    return y_pred


def bootstraped_score(y_true, y_pred, classes, score_func, score_func_params={}, n_bootstrap=100, random_seed=0):
    assert len(y_pred) == len(
        y_true), 'lenghts of y_pred and y_true should be the same'
    np.random.seed(random_seed)

    scores = []
    for i in range(n_bootstrap):
        resampled_y_true, resampled_y_pred = resample(
            y_true, y_pred, replace=True,
            n_samples=len(y_true), stratify=y_true,
            random_state=(random_seed+i))

        if resampled_y_pred.ndim > 1:
            presented_classes = np.unique(resampled_y_true)
            resampled_y_pred = leave_presented_classes(
                resampled_y_pred, classes, presented_classes
            )
            score_func_params.update({'labels': presented_classes})

        scores.append(score_func(
            resampled_y_true,
            resampled_y_pred,
            **score_func_params))

    return scores


def get_mask_count_greater(arr, min_count=1):
    u, counts = np.unique(arr, return_counts=True)
    return np.isin(arr, u[counts > min_count])


class CatBoostClassifierTrainTestSplit(CatBoostClassifier):
    def __init__(self, test_size=0.2, *args, **kwargs):
        self.test_size = test_size
        super().__init__(*args, **kwargs)

    def fit(self, x, y, **kwargs):
        train_x, eval_x, train_y, eval_y = train_test_split(
            x, y, test_size=self.test_size, stratify=y)
        super().fit(X=train_x, y=train_y, eval_set=(eval_x, eval_y))
        return self


def report_indirect_discrimination(gini_ci_by_feat, conf_lvl):
    df = pd.DataFrame()
    lower_bounds, upper_bounds = zip(*gini_ci_by_feat.values())
    is_check_feat = ['yes' if ci_u >
                     0.5 else 'no' for ci_u in upper_bounds]
    df['Защищенная характеристика'] = list(gini_ci_by_feat.keys())
    df[f'Левая граница CI_{conf_lvl} Gini'] = lower_bounds
    df[f'Правая граница CI_{conf_lvl} Gini'] = upper_bounds
    df['Защищенная характеристика не проверяется на дискриминацию'] = is_check_feat
    df['Результат теста'] = [
        'red' if i == 'yes' else 'green' for i in is_check_feat]

    result = {
        'semaphore': 'gray',
        'result_dataframes': [df],
        'result_plots': [],
        'result_dict': None
    }
    return result


def test_indirect_discrimination_gini(sampler, protected_feats: list,
                                      conf_lvl=0.95, n_bootstrap=200,
                                      feature_model=None, feature_preprocessor=None,
                                      cat_features=None, random_seed=1, n_jobs=1, precomputed=None, **kwargs):
    x_fair_train = pd.DataFrame(
        sampler.train['X']).drop(columns=protected_feats)
    x_fair_test = pd.DataFrame(sampler.oos['X']).drop(columns=protected_feats)
    if feature_model is None:
        if cat_features is None:
            cat_features = get_cat_features(x_fair_train)
        cat_features = list(set(cat_features).difference(protected_feats))
        feature_model = CatBoostClassifierTrainTestSplit(iterations=64,
                                                         cat_features=cat_features,
                                                         verbose=0,
                                                         random_seed=random_seed,
                                                         thread_count=n_jobs,
                                                         eval_metric='AUC',
                                                         od_wait=16)
    if feature_preprocessor is None:
        feature_preprocessor = Pipeline(steps=(
            ('nan', SimpleImputer(strategy='constant', fill_value=1e10)),
            ('quantization', Quantizer())))

    gini_quantiles = dict()
    for protected_feat in protected_feats:
        fair_target_train = interface.get_columns(
            sampler.train['X'], [protected_feat])
        fair_target_train = feature_preprocessor.fit_transform(
            fair_target_train)
        fair_target_train = np.ravel(fair_target_train)
        fair_target_test = interface.get_columns(
            sampler.oos['X'], [protected_feat])
        fair_target_test = feature_preprocessor.transform(fair_target_test)
        fair_target_test = np.ravel(fair_target_test)

        fair_pred_test = feature_model.fit(
            x_fair_train, fair_target_train).predict_proba(x_fair_test)
        if fair_pred_test.shape[1] == 2:
            fair_pred_test = fair_pred_test[:, 1]
            score_func_params = {}
        else:
            score_func_params = {
                'multi_class': 'ovr'
            }

        gini_scores = bootstraped_score(fair_target_test, fair_pred_test, feature_model.classes_,
                                        gini_score, score_func_params=score_func_params,
                                        n_bootstrap=n_bootstrap)
        ci_l = np.quantile(gini_scores, q=(1 - conf_lvl) / 2)
        ci_u = np.quantile(gini_scores, q=(1 + conf_lvl) / 2)
        gini_quantiles[protected_feat] = (ci_l, ci_u)
    result = report_indirect_discrimination(gini_quantiles, conf_lvl)
    if precomputed is not None:
        # for test_delete_protected
        precomputed['result_test_indirect_discrimination_gini'] = result
    return result
