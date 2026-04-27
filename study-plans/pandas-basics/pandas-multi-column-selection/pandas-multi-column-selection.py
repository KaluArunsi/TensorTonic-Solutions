import pandas as pd

def select_columns(data, columns):
    """
    Returns: dict mapping selected column names to value lists
    """
    df = pd.DataFrame(data)

    cols = df[columns]
    col_dict = cols.to_dict(orient='list')

    return col_dict