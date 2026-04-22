import pandas as pd

def head_tail(data, n):
    """
    Returns: dict with 'head' and 'tail' (both dicts of column -> list)
    """
    df = pd.DataFrame(data)

    h_t = {
        'head': {key: [*val] for key, val in df.head(n).items()},
        'tail': {key: [*val] for key, val in df.tail(n).items()}
    }

    return h_t