import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    df_list = [pd.DataFrame(dict) for dict in dfs]

    result = pd.concat(df_list)

    return [[result.shape[0], result.shape[1]], result.to_dict(orient='list')]