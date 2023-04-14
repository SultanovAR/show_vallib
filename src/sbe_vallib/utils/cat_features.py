import pandas as pd


def get_cat_features(data, less_count_is_cat=20):
    df = pd.DataFrame(data)
    cat_features = set(df.select_dtypes(
        include=['object', 'category']).columns)
    low_unique = set(df.columns[df.nunique() <= less_count_is_cat])
    cat_features = list(cat_features | low_unique)
    return cat_features
