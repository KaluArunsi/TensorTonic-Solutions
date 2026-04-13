import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    q_kt = torch.matmul(Q, K.transpose(-2, -1))

    root_dk = math.sqrt(K.shape[-1]) #d_k is the last item in K

    scores = q_kt / root_dk

    softmax = F.softmax(scores, dim=-1)

    attention = torch.matmul(softmax, V)

    return attention