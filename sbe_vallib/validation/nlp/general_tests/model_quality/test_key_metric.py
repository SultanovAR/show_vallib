import pandas as pd
import numpy as np


def test_key_metric(
    model,
    scorer,
    sampler,
    n_iter=200,
    refit=False
):

    metrics = []
    for i in range(n_iter):
        sampler.set_seed(i, bootstrap=True)
        train = sampler.train
        oos = sampler.oos
        if refit:
            model.fit(X=train["X"], y=train["y_true"])
        else:
            y_pred = model.predict(oos["X"])
        metrics.append(scorer.score(oos["y_true"], y_pred))
        
    return {
        "semaphore": "grey",
        "result_dict": {"metrics": metrics},
        "result_dataframes": [pd.DataFrame(metrics)],
        "result_plots": [],
    }
