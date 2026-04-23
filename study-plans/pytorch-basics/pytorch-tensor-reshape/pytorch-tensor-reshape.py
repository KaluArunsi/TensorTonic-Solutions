import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)

    if op == 'flatten':
        return torch.flatten(x)
    elif op == 'squeeze':
        return x.squeeze()
    else:
        return x.T