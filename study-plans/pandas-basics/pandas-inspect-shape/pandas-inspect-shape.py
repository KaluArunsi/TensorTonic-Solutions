import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)
    inspect_dict = {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'columns': [*df.columns],
        'dtypes': {col: str(dt) for col, dt in df.dtypes.items()},
        'total_values': df.size
    }

    return inspect_dict