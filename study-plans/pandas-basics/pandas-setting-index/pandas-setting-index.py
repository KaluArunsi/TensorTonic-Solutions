import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    df = pd.DataFrame(data)
    index_values = [*df[index_col]]
    
    df = df.set_index(index_col)
    columns = [*df.columns]
    data = df.to_dict(orient='list')

    index_dict = {
        'index_values' : index_values,
        'columns' : columns,
        'data' : data
    }

    return index_dict