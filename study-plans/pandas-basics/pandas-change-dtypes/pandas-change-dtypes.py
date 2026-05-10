import pandas as pd

def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    df = pd.DataFrame(data)
    dtypes_before = df.dtypes.apply(lambda x : x.name).to_dict()

    df[column] = df[column].astype(target_type)
    dtypes_after = df.dtypes.apply(lambda x : x.name).to_dict()

    return [dtypes_before, dtypes_after]