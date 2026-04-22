import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    col_dict = {
        'values': [*df[column]],
        'length': len(df[column])
    }

    return col_dict