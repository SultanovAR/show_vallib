import pandas as pd
from sbe_vallib.sampler import SupervisedSampler


def test_ci(model, sampler: SupervisedSampler, scorer, n_iter=200, use_predict_proba=True, **kwargs):
    metrics = []
    for i in range(n_iter):
        sampler.set_state(seed=i, gen_method='bootstrap')
        train = sampler.train
        oos = sampler.oos
        if sampler._gen_method != 'bootstrap':
            model.fit(X=train["X"], y=train["y_true"])
        if use_predict_proba:
            y_pred = model.predict_proba(oos['X'])
        else:
            y_pred = model.predict(oos['X'])
        metrics.append(scorer.calc_metrics(oos['y_true'], y_pred))
    return {
        'semaphore': 'grey',
        'result_dict': {'metrics': metrics},
        'result_dataframes': [pd.DataFrame(metrics)],
        'result_plots': []
    }
