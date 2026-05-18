import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    mean = X.mean(dim=0)
    variance = X.var(dim=0, unbiased=False)

    norm_x = (X - mean) / torch.sqrt(variance + eps)

    batch_tensor = gamma * norm_x + beta

    return batch_tensor