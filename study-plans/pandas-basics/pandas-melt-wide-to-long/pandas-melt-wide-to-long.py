import pandas as pd

def melt_dataframe(data, id_vars, value_vars):
    """
    Returns: dict with keys from id_vars plus 'variable' and 'value'
    """
    df = pd.DataFrame(data)

    long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars
    )

    return long.to_dict(orient='list')