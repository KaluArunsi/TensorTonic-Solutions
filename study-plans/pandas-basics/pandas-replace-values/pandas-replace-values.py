import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)

    count_target = int((df[column] == old_val).sum())

    df[column].replace(old_val, new_val, inplace=True)

    return {
        'data': df.to_dict(orient='list'),
        'count': count_target
    }