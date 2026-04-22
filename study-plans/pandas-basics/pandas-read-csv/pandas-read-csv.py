import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df = pd.DataFrame(data)
    data_dict = {
        'data' : data,
        'shape' : [*df.shape],
        'columns' : [*df.columns]
    }

    return data_dict