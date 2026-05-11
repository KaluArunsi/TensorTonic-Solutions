import pandas as pd

def apply_transform(data, column, operation_type):
    """
    Returns: dict with original columns plus column_transformed
    """
    df = pd.DataFrame(data)
    target_series = df[column]

    # Mapping operation strings to their logic
    if operation_type == "normalize":
        # Min-Max Scaling [0, 1] rounded to 4 decimals
        diff = target_series.max() - target_series.min()
        if diff == 0:
            transformed = target_series * 0 # Avoid division by zero
        else:
            transformed = ((target_series - target_series.min()) / diff).round(4)
            
    elif operation_type == "rank":
        # Integer rank (1 = smallest)
        transformed = target_series.rank(method='min').astype(int)
        
    elif operation_type == "cumsum":
        # Cumulative sum
        transformed = target_series.cumsum()
        
    elif operation_type == "double":
        # Multiply each value by 2
        transformed = target_series * 2
    else:
        # Fallback for custom operations
        transformed = target_series.apply(operation_type)

    df[column + '_transformed'] = transformed
    return df.to_dict(orient='list')