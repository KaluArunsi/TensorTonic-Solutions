import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)

    if method == 'relu':
        return torch.clamp(x, min=0)
    elif method == 'sigmoid':
        return 1 / (1 + torch.exp(-x))
    elif method == 'tanh':
        plus_exp = torch.exp(x)
        minus_exp = torch.exp(-x)

        return (plus_exp - minus_exp) / (plus_exp + minus_exp)
    elif method == 'leaky_relu':
        return torch.where(x > 0, x, 0.01 * x)