import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    values = torch.tensor(values, dtype=torch.float32, requires_grad=True)

    y = sum(values**3 + 2*values)
    y_grad = y.backward()

    return values.grad.tolist()
