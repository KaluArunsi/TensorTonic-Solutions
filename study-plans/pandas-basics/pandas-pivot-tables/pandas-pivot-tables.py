import pandas as pd

def create_pivot(data, index, columns, values, aggfunc):
    """
    Returns: nested dict {column_value: {index_value: agg_result}}
    """
    df = pd.DataFrame(data)

    df_pivot = pd.pivot_table(
        df,
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc
    ).fillna(0.0)

    return df_pivot.to_dict()