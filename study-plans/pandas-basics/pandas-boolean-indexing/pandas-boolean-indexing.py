import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)

    filtered_df = df[df[column] > threshold]
    filtered_data = filtered_df.to_dict(orient='list')
    count = len(filtered_df)
    
    filter_dict = {
        "filtered_data" : filtered_data,
        "count" : count
    }

    return filter_dict