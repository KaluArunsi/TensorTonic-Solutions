import pandas as pd

def data_types_overview(data):
    """
    Returns: dict with 'dtypes', 'type_counts', 'num_columns'
    """
    df = pd.DataFrame(data)

    dtype_view = {
        'dtypes': {col: str(dt) for col, dt in df.dtypes.items()},
        'type_counts': df.dtypes.value_counts().rename(index=str).to_dict(),
        'num_columns': df.shape[1]
    }

    return dtype_view