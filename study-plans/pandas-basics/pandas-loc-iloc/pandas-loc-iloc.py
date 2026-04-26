import pandas as pd

def iloc_selection(data, row, col):
    """
    Returns: list [element, row_values, col_values]
    """
    df = pd.DataFrame(data)
    element = df.iloc[row, col]
    row_values = list(df.iloc[row])
    col_values = list(df.iloc[:, col])
    
    loc_list = [element, row_values, col_values]

    return loc_list