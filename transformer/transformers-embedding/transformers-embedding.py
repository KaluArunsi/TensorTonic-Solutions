import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedded_matrix = nn.Embedding(vocab_size, d_model)
    return embedded_matrix

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    scaled_dim = math.sqrt(d_model)

    look_up = embedding(tokens)

    big_E = look_up * scaled_dim

    return big_E