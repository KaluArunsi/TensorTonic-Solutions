import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data)

    sum_grouping = df.groupby(group_col)[value_col].sum()
    mean_grouping = sums = df.groupby(group_col)[value_col].mean()
    count_grouping = sums = df.groupby(group_col)[value_col].count()

    return {
        'sum' : sum_grouping.to_dict(),
        'mean' : mean_grouping.to_dict(),
        'count' : count_grouping.to_dict()
    }