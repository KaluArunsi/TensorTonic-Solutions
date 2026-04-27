import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    df = pd.DataFrame(data)
    df = df.set_index(index_col)
    
    cols_before = [*df.columns]
    df = df.reset_index()
    cols_after = [*df.columns]

    return [cols_before, cols_after]