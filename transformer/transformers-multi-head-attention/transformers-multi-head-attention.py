import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    q, k, v = np.dot(Q, W_q), np.dot(K, W_k), np.dot(V, W_v)

    projections = [q, k, v]
    reshaped_projections = []

    for x in projections:
        batch, seq, d_model = x.shape
        d_k = d_model // num_heads

        x = x.reshape(batch, seq, num_heads, d_k)
        x = x.transpose(0, 2, 1, 3) # batch, num_heads, seq, d_k
        reshaped_projections.append(x)

    new_q, new_k, new_v = reshaped_projections

    all_qkt = np.matmul(new_q, new_k.transpose(0, 1, 3, 2))
    root_dk = np.sqrt(d_k)

    scores = all_qkt / root_dk

    attn = np.matmul(softmax(scores), new_v)
    attn = attn.transpose(0, 2, 1, 3)

    attn_combined = attn.reshape(batch, seq, d_model)

    multi_head_attn = np.matmul(attn_combined, W_o)

    return multi_head_attn