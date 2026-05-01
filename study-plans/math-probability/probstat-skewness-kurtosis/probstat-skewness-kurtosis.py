import numpy as np

def skewness_kurtosis(data):
    """
    Returns: dict with 'skewness', 'kurtosis', and interpretation strings.
    Implementing the raw Adjusted Fisher formulas for Sample Skew and Kurtosis.
    """
    data = np.array(data, dtype='float64')
    n = data.size
    
    # Calculate components
    mu = np.mean(data)
    s = np.std(data, ddof=1)  # Sample standard deviation (s)
    
    # Standardized deviations: (xi - x_bar) / s
    z = (data - mu) / s
    
    # 1. Sample skewness (adjusted):
    # g1 = [n / ((n-1)(n-2))] * sum(z**3)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(z**3)
    
    # 2. Sample excess kurtosis (adjusted):
    # g2 = [n(n+1) / ((n-1)(n-2)(n-3))] * sum(z**4) - [3(n-1)^2 / ((n-2)(n-3))]
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2 = (3 * (n - 1)**2) / ((n - 2) * (n - 3))
    kurtosis = term1 * np.sum(z**4) - term2

    # Round numeric values to 4 decimal places
    skew_val = round(float(skew), 4)
    kurt_val = round(float(kurtosis), 4)

    # Classification logic
    if skew_val > 0.5:
        s_int = "right-skewed"
    elif skew_val < -0.5:
        s_int = "left-skewed"
    else:
        s_int = "approximately symmetric"

    if kurt_val > 1:
        k_int = "leptokurtic"
    elif kurt_val < -1:
        k_int = "platykurtic"
    else:
        k_int = "mesokurtic"

    return {
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'skew_interpretation': s_int,
        'kurtosis_interpretation': k_int
    }