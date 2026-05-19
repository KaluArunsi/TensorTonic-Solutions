import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)
    last_avg_grad = None

    for i, (x, t) in enumerate(micro_batches):
        x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
        
        pred = torch.dot(w, x)
        loss = ((pred - t) ** 2) / accum_steps

        loss.backward()

        if (i + 1) % accum_steps == 0:
            last_avg_grad = w.grad.clone()

            with torch.no_grad():
                w -= lr * w.grad

            w.grad.zero_()

    return w.detach().tolist(), last_avg_grad.detach().tolist()