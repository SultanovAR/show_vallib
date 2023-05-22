from collections import defaultdict

import pandas as pd
import numpy as np


def get_represent_value(data, is_categorical: bool = False):
    if is_categorical:
        return pd.Series(data).mode().values[0]
    return np.mean(data)


def get_swapped_oppr_priv_predictions(mask_oppressed_privileged,
                                      model,
                                      x_test, y_test,
                                      cat_features):
    """
    Collects target, returns source predictions and predictions after swapping for
    oppressed and privileaged groups for each protected feature.
    Swapped predictions for a particular feature_1 - are predictions
    when the values of feature_1 were swapped between oppressed
    and privileaged groups

    Parameters
    ----------
    model
        a model with a 'predict_proba' method
    mask_oppressed_privileged
        Dict with the following format
        {
            'feature': {
                'oppr': {'mask': array, 'representative_value': str, 'value': value},
                'priv': {'mask': array, 'representative_value': str, 'value': value}
            }
        }
    x_test
        features from the test dataset
    y_test
        target from the test dataset
    protected_feats: List[str]
        features for which discrimination tests will be provided
    cat_features: List[str]
        list of categorical features

    Returns
    -------
    Dict with the format:
    {"feature":
        {"source_preds": list (for privileaged and oppresed rows),
         "swaped_preds": list (for privileaged and oppresed rows),
         "target": list (for privileaged and oppresed rows)
    }}
    """
    swaped_preds_by_feature = defaultdict(dict)
    for protected_feat in mask_oppressed_privileged:
        oppr_priv = mask_oppressed_privileged[protected_feat]
        swap_value = dict()
        swap_value['priv'] = get_represent_value(x_test.loc[oppr_priv['oppr']['mask'], protected_feat],
                                                 bool(protected_feat in cat_features))
        swap_value['oppr'] = get_represent_value(x_test.loc[oppr_priv['priv']['mask'], protected_feat],
                                                 bool(protected_feat in cat_features))

        for group_type in ('oppr', 'priv'):
            data = x_test.loc[oppr_priv[group_type]['mask']]
            swaped_data = data.copy()
            swaped_data[protected_feat] = swap_value[group_type]
            preds = model.predict_proba(data)[:, 1]
            swaped_preds = model.predict_proba(swaped_data)[:, 1]

            swaped_preds_by_feature[protected_feat].update({
                group_type: {
                    'source_preds': preds,
                    'swapped_preds': swaped_preds,
                    'target': y_test.loc[oppr_priv[group_type]['mask']]
                }
            })
    return swaped_preds_by_feature
