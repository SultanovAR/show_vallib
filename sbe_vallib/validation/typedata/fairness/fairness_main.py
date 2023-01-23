from typing import List, Union, Dict
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tqdm import tqdm

from utils import gini_conf_interval, gini, bootstraped_score, shuffle_data
from stdlib_extensions import Quantization, CatBoostClassifierTrainTestSplit, Interval


class FairnessValidation():

    def __init__(self,
                 model,
                 train: Dict[str, Union[pd.DataFrame, pd.Series]],
                 oos: Dict[str, Union[pd.DataFrame, pd.Series]],
                 cat_features: List[str],
                 protected_feats: List[str],
                 n_unique_continious: int = 20,
                 min_perc_value: float = 0.01,
                 min_amount_pos: int = 10,
                 n_bootstrap: int = 200,
                 random_seed: int = 0,
                 n_jobs: int = 1):

        self.train, self.oos = dict(), dict()
        self.train['x'], self.train['y'] = shuffle_data(train['x'], train['y'])
        self.oos['x'], self.oos['y'] = shuffle_data(oos['x'], oos['y'])

        self.model = model
        self.cat_features = cat_features
        self.protected_feats = protected_feats
        self.n_unique_continious = n_unique_continious
        self.n_bootstrap = 200
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.min_perc_value = min_perc_value
        self.min_amount_pos = min_amount_pos
        self.test_list = [self.test_indirect_discrimination,
                          self.test_target_rate_delta,
                          self.test_tprd_fprd_delta,
                          self.test_oppressed_privileged_ci]

    def indirect_discrimination_gini(self,
                                     train: pd.DataFrame,
                                     test: pd.DataFrame,
                                     protected_feats: List[str],
                                     feat_model,
                                     feat_preproc,
                                     n_unique_continious: int = 20,
                                     confidence_level=0.95,
                                     n_bootstrap=200):
        x_train = train.drop(columns=protected_feats)
        x_test = test.drop(columns=protected_feats)

        gini_quantiles = dict()
        for protected_feat in protected_feats:
            y_train = train[protected_feat].to_numpy()
            y_test = test[protected_feat].to_numpy()
            if (is_numeric_dtype(train[protected_feat]) and
                    len(np.unique(train[protected_feat])) > n_unique_continious):
                y_train = feat_preproc.fit_transform(
                    train[protected_feat][:, None]).flatten()
                y_test = feat_preproc.transform(
                    test[protected_feat][:, None]).flatten()

            y_test_pred = feat_model.fit(
                x_train, y_train).predict_proba(x_test)
            if len(np.unique(y_test)) > 20:
                y_test[y_test > 20.1] = 20
                y_test[y_test < 0.9] = 1
            if y_test_pred.shape[1] == 2:
                y_test_pred = y_test_pred[:, 1]
                gini_scores = bootstraped_score(
                    y_test, y_test_pred, gini, n_bootstrap=n_bootstrap)
            else:
                gini_scores = bootstraped_score(y_test, y_test_pred,
                                                partial(gini, multi_class='ovr'), n_bootstrap=n_bootstrap)
            ci_l = np.quantile(gini_scores, q=(1 - confidence_level) / 2)
            ci_u = np.quantile(gini_scores, q=(1 + confidence_level) / 2)
            gini_quantiles[protected_feat] = (ci_l, ci_u)
        return gini_quantiles

    def report_indirect_discrimination(self, gini_ci_by_feat, confidence_level):
        report = pd.DataFrame()
        lower_bounds, upper_bounds = zip(*gini_ci_by_feat.values())
        is_check_feat = ['yes' if ci_u >
                         0.5 else 'no' for ci_u in upper_bounds]
        report['Защищенная характеристика'] = list(gini_ci_by_feat.keys())
        report[f'Левая граница CI_{confidence_level} Gini'] = lower_bounds
        report[f'Правая граница CI_{confidence_level} Gini'] = upper_bounds
        report['Защищенная характеристика не проверяется на дискриминацию'] = is_check_feat
        return report

    def test_indirect_discrimination(self, confidence_level=0.95):
        current_cat_features = list(
            set(self.cat_features) - set(self.protected_feats))
        feat_preproc = Pipeline((('nan', SimpleImputer(strategy='constant', fill_value=np.inf)),
                                ('quantization', Quantization())))
        feat_model = CatBoostClassifierTrainTestSplit(iterations=128,
                                                      cat_features=current_cat_features,
                                                      verbose=False,
                                                      random_seed=self.random_seed,
                                                      thread_count=self.n_jobs,
                                                      eval_metric='AUC',
                                                      od_wait=16)

        gini_ci_by_feat = self.indirect_discrimination_gini(self.train['x'],
                                                            self.oos['x'],
                                                            self.protected_feats,
                                                            feat_preproc=feat_preproc,
                                                            feat_model=feat_model,
                                                            n_unique_continious=self.n_unique_continious,
                                                            confidence_level=confidence_level,
                                                            n_bootstrap=self.n_bootstrap)
        result = self.report_indirect_discrimination(
            gini_ci_by_feat, confidence_level)
        return result

    def get_oppressed_privileged(self,
                                 x_train: pd.DataFrame,
                                 x_test: pd.DataFrame,
                                 y_test: Union[np.ndarray, pd.Series],
                                 protected_feats: List[str],
                                 feat_preproc,
                                 min_freq_value=0.01,
                                 min_amount_pos=50):
        """Defines oppressed and privileaged groups for each feature in protected_feats.
        an oppressed group - value of feature for which the target rate is minimal.
        a privileaged group - value of feature for which the target rate is maximal.
        Values of feature is grouping with quantizer

        Parameters
        ----------
        x_train : pd.DataFrame
            features from the train dataset
        x_test : pd.DataFrame
            features from the test dataset
        y_test : Union[np.ndarray, pd.Series]
            target from the test dataset
        protected_feats : List[str]
            features for which discrimination tests will be provided
        feat_preproc : _type_
            preprocessor in sklearn format (with fit() and transform() methods)
            it discretizises/(imputes NaNs) features that are continues
            or have a lot of unique values.
        min_freq_value : float, optional
            a value can represent a privileged or oppressed group only if
            freq(value) > min_freq_value, by default 0.01
        min_amount_pos : int, optional
            a value can represent a privileged or oppressed group only if
            target values has enough positive labels, by default 50

        Returns
        -------
        _type_
            _description_
        """
        mask_of_oppressed_privileged = defaultdict(dict)
        for feat in protected_feats:
            # preprocessing
            feature_df = pd.DataFrame(
                {feat: x_test[feat], 'target': y_test})
            mask = np.array(feature_df.notna().prod(axis=1), dtype=bool)
            feat_preproc.fit(x_train[feat].values[:, None])
            feature_df.loc[mask, feat] = feat_preproc.transform(
                feature_df.loc[mask, feat].values[:, None]).flatten()

            # filtering
            num_positive = feature_df.loc[mask, :].groupby(by=feat)[
                'target'].sum()
            num_values = feature_df.loc[mask, feat].nunique()
            if num_values > 2:
                counts = feature_df.loc[mask, feat].value_counts()
                enough_samples = ((counts / num_values) > min_freq_value).index
                enough_positive = num_positive[num_positive >
                                               min_amount_pos].index
                well_presented_values = set(
                    enough_samples) & set(enough_positive)
                mask &= feature_df[feat].isin(well_presented_values)

            #oppressed and privileaged
            grouped_target = feature_df.loc[mask, :].groupby(by=feat)[
                'target'].mean()
            for group_type, value in zip(('oppr', 'priv'),
                                         (grouped_target.idxmin(), grouped_target.idxmax())):
                feat_mask = mask & (feature_df[feat] == value)
                mask_of_oppressed_privileged[feat].update(
                    {group_type: np.array(feat_mask, dtype=bool)})

        return mask_of_oppressed_privileged

    def oppressed_privileged_ci(self,
                                x_train: pd.DataFrame,
                                x_test: pd.DataFrame,
                                y_test: Union[np.ndarray, pd.Series],
                                y_test_preds: Union[np.ndarray, pd.Series],
                                protected_feats: List[str],
                                feat_preproc,
                                confidence_levels=(0.95, 0.99),
                                min_freq_value=0.01,
                                min_amount_pos=50):
        """Measures gini confidence intervals for oppressed and privileaged groups

        Parameters
        ----------
        x_train : pd.DataFrame
            features from the train dataset
        x_test : pd.DataFrame
            features from the test dataset
        y_test : Union[np.ndarray, pd.Series]
            target from the test dataset
        y_test_preds : Union[np.ndarray, pd.Series]
            estimation of positive class from model 
        protected_feats : List[str]
            features for which discrimination tests will be provided
        feat_preproc : _type_
            preprocessor in sklearn format (with fit() and transform() methods)
            it discretizises/(imputes NaNs) features that are continues
            or have a lot of unique values.
        confidence_levels : tuple, optional
            a tuple of confidence levels for whom confidence intervals
            will be computed, by default (0.95, 0.99)
        min_freq_value : float, optional
            a value can represent a privileged or oppressed group only if
            freq(value) > min_freq_value, by default 0.01
        min_amount_pos : int, optional
            a value can represent a privileged or oppressed group only if
            target values has enough positive labels, by default 50

        Returns
        -------
        Dict with the format:
        {"feature":
            "oppr": {ci_lvl: (ci_l, ci_r)}
            "priv": {ci_lvl: (ci_l, ci_r)}
        }
        """

        mask_oppressed_privileged = self.get_oppressed_privileged(x_train,
                                                                  x_test,
                                                                  y_test,
                                                                  protected_feats,
                                                                  feat_preproc,
                                                                  min_freq_value,
                                                                  min_amount_pos)
        ci_by_feature = dict()
        for protected_feat in mask_oppressed_privileged:
            for group_type in ('oppr', 'priv'):
                group_mask = mask_oppressed_privileged[protected_feat][group_type]
                group_target = pd.Series(y_test)[group_mask]
                group_score = pd.Series(y_test_preds)[group_mask]

                ci_by_level = dict()
                for ci_lvl in confidence_levels:
                    if (group_target.mean() == 0.0) or (group_target.mean() == 1.0):
                        ci = 'невозможно расчитать, в группе один класс'
                    else:
                        ci = gini_conf_interval(
                            group_score, group_target, alpha=ci_lvl)
                    ci_by_level.update({ci_lvl: ci})
                ci_by_feature.setdefault(protected_feat, {}).update({
                    group_type: ci_by_level
                })

        return ci_by_feature

    def report_oppressed_privileged(self, oppressed_privileged_ci, conf_lvls=(0.95, 0.99)):
        """
        Creates a report for the results of 'test_oppressed_privileged_ci'
        in pandas DataFrame format

        Parameters
        ----------
        oppressed_privileged_ci: Dict
            Dict with the format: 
            {"feature":
                "oppr": {ci_lvl: (ci_l, ci_r)}
                "priv": {ci_lvl: (ci_l, ci_r)}
            }   

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            f'{conf_lvl * 100}%-ый доверительный интервал для угнетаемой группы'
            f'{conf_lvl * 100}%-ый доверительный интервал для привелегированной группы'
            'Результат теста 4'
        """
        def color_criteria(oppr, priv, conf_lvls=(0.95, 0.99)):
            for group in (oppr, priv):
                for ci_lvl in conf_lvls:
                    if isinstance(group[ci_lvl], str):
                        return 'gray'
                    group[ci_lvl] = Interval(*group[ci_lvl])

            narrow = min(conf_lvls)
            wide = max(conf_lvls)
            color = 'red'
            if ((oppr[wide].left > 0.2) and (priv[wide].left > 0.2))\
                    or (not (oppr[wide] & priv[wide]).is_empty):
                color = 'yellow'
            if ((oppr[wide].left > 0.4) and (priv[wide].left > 0.4))\
                    or (not (oppr[narrow] & priv[narrow]).is_empty):
                color = 'green'
            if (oppr[narrow].length > 1) or (priv[narrow].length > 1):
                color = 'gray'
            return color

        report = defaultdict(list)
        report['Защищенная характеристика'] = list(
            oppressed_privileged_ci.keys())
        for feat in oppressed_privileged_ci.keys():
            oppr = oppressed_privileged_ci[feat]['oppr']
            priv = oppressed_privileged_ci[feat]['priv']
            for conf_lvl in conf_lvls:
                report[f'{conf_lvl * 100}%-ый доверительный интервал для угнетаемой группы'].append(
                    oppr[conf_lvl])
                report[f'{conf_lvl * 100}%-ый доверительный интервал для привелегированной группы'].append(
                    priv[conf_lvl])
            report['Результат теста 4'].append(
                color_criteria(oppr, priv, conf_lvls))
        return pd.DataFrame(report)

    def test_oppressed_privileged_ci(self):
        """
        "Часть 31. Тест 7.4. Сравнение ранжирующей способности внутри защищенной
        характеристики". Test measures gini's confidence interval under oppressed
        and privileaged groups. After that it compares confidence intervals.

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            f'{conf_lvl * 100}%-ый доверительный интервал для угнетаемой группы'
            f'{conf_lvl * 100}%-ый доверительный интервал для привелегированной группы'
            'Результат теста 4'
        """
        assert self.oos['y'].nunique(
        ) == 2, 'test can work only with binary models'

        feat_preproc = Pipeline([('nan', SimpleImputer(strategy='constant', fill_value=np.inf)),
                                 ('quantization', Quantization())])
        scores = self.model.predict_proba(self.oos['x'])[:, 1]
        oppr_priv_ci = self.oppressed_privileged_ci(x_train=self.train['x'],
                                                    x_test=self.oos['x'],
                                                    y_test=self.oos['y'],
                                                    y_test_preds=scores,
                                                    protected_feats=self.protected_feats,
                                                    feat_preproc=feat_preproc,
                                                    confidence_levels=(
                                                        0.95, 0.99),
                                                    min_freq_value=0.01,
                                                    min_amount_pos=50)
        result = self.report_oppressed_privileged(oppr_priv_ci)
        return result

    def get_represent_value(self, data, is_categorical: bool = False):
        if is_categorical:
            return pd.Series(data).mode().values[0]
        return np.mean(data)

    def swaped_oppr_priv_predictions(self, model, x_train, x_test,
                                     y_test, protected_feats, cat_features,
                                     feat_preproc, min_freq_value, min_amount_pos):
        """
        Collects target, swap predictions and source predictions for
        oppressed and privileaged groups for each protected feature.
        Swap predictions for a particular feature_1 - are predictions
        when the values of feature_1 were swapped between oppressed
        and privileaged groups

        Parameters
        ----------
        model
            a model with a 'predict_proba' method
        x_train
            features from the train dataset
        x_test
            features from the test dataset
        y_test
            target from the test dataset
        protected_feats: List[str]
            features for which discrimination tests will be provided
        cat_features: List[str]
            list of categorical features
        feat_preproc
            preprocessor in sklearn format (with fit() and transform() methods)
            it discretizises/(imputes NaNs) features that are continues
            or have a lot of unique values.
        min_freq_value:
            a value can represent a privileged or oppressed group only if
            freq(value) > min_freq_value.
        min_amount_pos:
            a value can represent a privileged or oppressed group only if
            target values has enough positive labels.

        Returns
        -------
        Dict with the format:
        {"feature":
            {"preds": list,
             "swaped_preds": list,
             "target": list}}
        """
        mask_oppressed_privileged = self.get_oppressed_privileged(x_train=x_train,
                                                                  x_test=x_test,
                                                                  y_test=y_test,
                                                                  protected_feats=protected_feats,
                                                                  feat_preproc=feat_preproc,
                                                                  min_freq_value=min_freq_value,
                                                                  min_amount_pos=min_amount_pos)
        swaped_preds_by_feature = defaultdict(dict)
        for protected_feat in mask_oppressed_privileged:
            mask = mask_oppressed_privileged[protected_feat]
            swap_value = dict()
            swap_value['priv'] = self.get_represent_value(x_test.loc[mask['oppr'], protected_feat],
                                                          bool(protected_feat in cat_features))
            swap_value['oppr'] = self.get_represent_value(x_test.loc[mask['priv'], protected_feat],
                                                          bool(protected_feat in cat_features))

            for group_type in ('oppr', 'priv'):
                data = x_test.loc[mask[group_type]]
                swaped_data = data.copy()
                swaped_data[protected_feat] = swap_value[group_type]
                preds = model.predict_proba(data)[:, 1]
                swaped_preds = model.predict_proba(swaped_data)[:, 1]

                swaped_preds_by_feature[protected_feat].update({
                    group_type: {
                        'source_preds': preds,
                        'swaped_preds': swaped_preds,
                        'target': y_test.loc[mask[group_type]]
                    }
                })
        return swaped_preds_by_feature

    def report_target_rate_delta(self, tr_delta_by_feat: dict):
        """
        Creates a report for the results of 'test_target_rate_delta' in pandas DataFrame format

        Parameters
        ----------
        tr_delta_by_feat: Dict
            Dict with the format: {"feature": {"oppr": value, "priv": value}}

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            'Минимальное относительное изменение прогнозного значения'
            'Максимальное относительное изменение прогнозного значения'
            'Результат теста'
        """
        def color_criteria(tr_delta_first, tr_delta_second):
            if max(abs(tr_delta_first), abs(tr_delta_second)) > 0.3:
                if tr_delta_first * tr_delta_first < 0:
                    return 'red'
                else:
                    return 'yellow'
            return 'green'

        report = pd.DataFrame()
        report['Защищенная характеристика'] = list(tr_delta_by_feat.keys())
        report['Минимальное относительное изменение прогнозного значения'] = [
            min(i) for i in tr_delta_by_feat.values()]
        report['Максимальное относительное изменение прогнозного значения'] = [
            max(i) for i in tr_delta_by_feat.values()]
        report['Результат теста 2'] = [
            color_criteria(i[0], i[1]) for i in tr_delta_by_feat.values()]
        return report

    def test_target_rate_delta(self):
        """
        "Часть 31. Тест 7.2. Сравнение прогнозного уровня целевой переменной
        при изменении защищенной характеристики". Test measures Target-Rate in predictions
        when feature values for oppressed and privileaged groups are swapped.

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            'Изменение TPRD после перестановки значений'
            'Изменение FPRD после перестановки значений'
            'Результат теста'
        """
        feat_preproc = Quantization()
        swaped_preds = self.swaped_oppr_priv_predictions(model=self.model,
                                                         x_train=self.train['x'],
                                                         x_test=self.oos['x'],
                                                         y_test=self.oos['y'],
                                                         protected_feats=self.protected_feats,
                                                         cat_features=self.cat_features,
                                                         feat_preproc=feat_preproc,
                                                         min_freq_value=self.min_perc_value,
                                                         min_amount_pos=self.min_amount_pos)
        cutoff = self.train['y'].mean()
        target_rates_delta_by_feat = defaultdict(list)
        for feat in swaped_preds:
            for group_type in swaped_preds[feat]:
                preds = np.array(
                    swaped_preds[feat][group_type]['source_preds'] > cutoff, dtype=int)
                swap_preds = np.array(
                    swaped_preds[feat][group_type]['swaped_preds'] > cutoff, dtype=int)
                tr_delta = (swap_preds.mean() - preds.mean()) / preds.mean()
                target_rates_delta_by_feat[feat].append(tr_delta)

        result = self.report_target_rate_delta(target_rates_delta_by_feat)
        return result

    def report_tprd_fprd_delta(self, tprd_fprd_delta_by_feat: Dict, thresholds=(0.1, 0.2)):
        """
        Creates a report for the results of 'test_tprd_fprd_delta' in pandas DataFrame format

        Parameters
        ----------
        tprd_fprd_delta_by_feat: Dict[Dict[str]]
            Dict with the format: {"feature": {"tprd": value, "fprd": value }}
        thresholds: Tuple
            Tuple with the format: (yellow_threshold, red_threshold)

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            'Изменение TPRD после перестановки значений'
            'Изменение FPRD после перестановки значений'
            'Результат теста'
        """
        def color_criteria(tprd, fprd):
            diff = max(tprd, fprd)
            if diff <= min(thresholds):
                return 'green'
            elif min(thresholds) < diff <= max(thresholds):
                return 'yellow'
            return 'red'

        report = pd.DataFrame()
        report['Защищенная характеристика'] = list(
            tprd_fprd_delta_by_feat.keys())
        report['Изменение TPRD после перестановки значений'] = [i['tprd']
                                                                for i in tprd_fprd_delta_by_feat.values()]
        report['Изменение FPRD после перестановки значений'] = [i['fprd']
                                                                for i in tprd_fprd_delta_by_feat.values()]
        report['Результат теста 3'] = [
            color_criteria(i['tprd'], i['fprd']) for i in tprd_fprd_delta_by_feat.values()]
        return report

    def test_tprd_fprd_delta(self):
        """
        "Часть 31. Тест 7.3. Анализ динамики TPR и FPR при изменении защищенной характеристики".
        Test measures True-Positive-Rate-Difference and False-Positive-Rate-Difference
        when feature values for oppressed and privileaged groups are swapped.

        Returns
        -------
        pd.DataFrame with columns:
            'Защищенная характеристика'
            'Изменение TPRD после перестановки значений'
            'Изменение FPRD после перестановки значений'
            'Результат теста'
        """
        feat_preproc = Quantization()
        swaped_preds = self.swaped_oppr_priv_predictions(model=self.model,
                                                         x_train=self.train['x'],
                                                         x_test=self.oos['x'],
                                                         y_test=self.oos['y'],
                                                         protected_feats=self.protected_feats,
                                                         cat_features=self.cat_features,
                                                         feat_preproc=feat_preproc,
                                                         min_freq_value=self.min_perc_value,
                                                         min_amount_pos=self.min_amount_pos)
        threshold = self.oos['y'].mean()
        tprd_fprd_delta_by_feat = defaultdict(dict)
        for feat in swaped_preds:
            groups_metric = {'tpr': [], 'fpr': [],
                             'swap_tpr': [], 'swap_fpr': []}
            for group_type in ('oppr', 'priv'):
                preds = np.array(
                    swaped_preds[feat][group_type]['source_preds'] > threshold, dtype=int)
                swap_preds = np.array(
                    swaped_preds[feat][group_type]['swaped_preds'] > threshold, dtype=int)
                target = np.array(
                    swaped_preds[feat][group_type]['target'], dtype=int)

                groups_metric['tpr'].append(preds[target == 1].mean())
                groups_metric['fpr'].append(preds[target == 0].mean())
                groups_metric['swap_tpr'].append(
                    swap_preds[target == 1].mean())
                groups_metric['swap_fpr'].append(
                    swap_preds[target == 0].mean())

            diff_metric = dict()
            for metric in groups_metric:
                diff_metric[metric] = max(
                    groups_metric[metric]) - min(groups_metric[metric])

            tprd_fprd_delta_by_feat[feat].update({
                'tprd': abs(diff_metric['swap_tpr'] - diff_metric['tpr']),
                'fprd': abs(diff_metric['swap_fpr'] - diff_metric['fpr'])
            })

        result = self.report_tprd_fprd_delta(tprd_fprd_delta_by_feat)
        return result

    def validate(self,):
        tests_result = dict()
        for test in tqdm(self.test_list):
            tests_result[test.__name__] = test()

        return tests_result
