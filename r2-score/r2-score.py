import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    edge_1 = np.all(y_true == y_true[0])
    edge_2 = np.all(y_pred == y_true[0])

    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - np.mean(y_true))**2) + 1e-15

    r2 = 1 - (sse / sst)
    r2 = float(r2)

    score = np.where(edge_1, np.where(edge_2, 1.0, 0.0), r2)

    return float(score)