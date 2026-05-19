import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    row_max = torch.max(logits, dim=1, keepdim=True)[0]
    exp_tensor = torch.exp(logits - row_max)
    exp_sum = torch.sum(exp_tensor, dim=1, keepdim=True)

    softmax = exp_tensor / exp_sum

    return softmax