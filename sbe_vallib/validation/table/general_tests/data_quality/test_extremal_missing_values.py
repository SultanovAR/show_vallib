import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy import stats
import sys, os

from sbe_vallib.validation.utility import make_test_report

def extremal_missing_values_test(train, 
                                feature_names = None, 
                                spec_value_share=0.1,
                                coef=1.5,
                                dist_names =['beta','chi2','expon','f','gamma','lognorm','logistic','norm','t','weibull_min'],
                                **params):
                                
    X_train = train
    spec_value_share = params.get('spec_value_share', spec_value_share)
    coef = params.get('coef_extremal_missing_values_test', coef)
    dist_names = params.get('dist_names', dist_names)
    feature_names = params.get('feature_names', feature_names)

    feature_names = feature_names if feature_names is not None else X_train.columns
    

    result = []
    for feature in tqdm(feature_names, leave = True, desc = 'extremal_missing_values_test: '):
        nan_values_index = X_train.index[X_train[feature].isnull()]
        extr_share = X_train[feature].dtype

        if X_train[feature].dtype in [np.int64, np.float64]: # BAG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            lower_bound, upper_bound = outlier_bounds(X_train[feature], spec_value_share=spec_value_share,coef=coef, dist_names=dist_names )
            extremal_values = (X_train[feature] < lower_bound) | (X_train[feature] > upper_bound)
            extr_share = extremal_values.sum() / len(X_train)
        res = {'Переменная': feature,
               'Доля пропущенных значений': len(nan_values_index) / len(X_train),
               'Доля экстремальных значений': extr_share}
        result.append(res)

    result = pd.DataFrame(result)
    #result_test = {'table':result, 'color':None, 'picture':None} #  то, что вывожу я
    result_test = make_test_report(title = 'extremal_missing_values_test', table = result)
    return result_test


    
def mle_dist(X, dist_name=None, limit_obs=100000):
    if len(X) > limit_obs:
        X = X.sample(limit_obs)
    else:
        X = X.copy()

    x_mean = -np.nanmean(X)
    X += x_mean  # transformation

    distr = getattr(stats, dist_name)
    try:
        params = distr.fit(X)
    except RuntimeError:
        return -np.inf, x_mean, (np.nan, np.nan)
    LL = distr.logpdf(X, *params).sum()

    X -= x_mean  # back transformation
    return LL, x_mean, params


def mle(X, dist_names):
    res = [mle_dist(X, name) for name in dist_names]
    LLs = list(zip(*res))[0]
    ind_max = np.argmax(LLs)
    best_mle, x_add, params = res[ind_max]
    best_name = dist_names[ind_max]
    return {
        'best_mle': best_mle,
        'best_name': best_name,
        'x_add': x_add,
        'params': params
    }


def cnvrt_dist(X, x_add=0, dist_name=None, params=None, coef=1.5):
    # transformation additive
    X += x_add

    dist = getattr(stats, dist_name)
    p_fit = dist.cdf(X, *params)
    q_fit = stats.norm.ppf(p_fit, 0, 1)
    q_25, q_75 = stats.scoreatpercentile(q_fit, [25, 75])
    iqr = q_75 - q_25
    lwr_bnd = q_25 - coef * iqr
    upr_bnd = q_75 + coef * iqr
    p_lrw_upr = stats.norm.cdf((lwr_bnd, upr_bnd), 0, 1)

    # back transformation
    q_lrw_upr = dist.ppf(p_lrw_upr, *params)

    # back mean_to_zero or non_negative
    final_lwr_bnd = q_lrw_upr[0] - x_add
    final_upr_bnd = q_lrw_upr[1] - x_add

    # back transformation additive
    X -= x_add
    return final_lwr_bnd, final_upr_bnd


def outlier_bounds(X, spec_value_share=0.1, 
        coef=1.5,
        dist_names = ['beta','chi2','expon','f','gamma','lognorm','logistic','norm','t','weibull_min']):
    X = pd.Series(X)
    vc = X.value_counts(normalize=True)
    spec_values = vc[vc >= spec_value_share].index.values
    filtered_sample = X.loc[X.notnull() & ~X.isin(spec_values)]
    if len(filtered_sample) == 0:
        return -np.inf, np.inf

    res = mle(filtered_sample, dist_names)
    lwr, upr = cnvrt_dist(filtered_sample,
                          x_add=res['x_add'],
                          dist_name=res['best_name'],
                          params=res['params'],
                          coef=coef)

    return lwr, upr






