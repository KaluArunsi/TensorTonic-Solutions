import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    rows_before = df.shape[0]

    df_clean = df.drop_duplicates()
    rows_after = df_clean.shape[0]

    return [
        rows_before,
        rows_after,
        df_clean.to_dict(orient='list')
    ]