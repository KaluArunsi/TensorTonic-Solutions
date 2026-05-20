import torch
import torch.nn as nn

def manual_train_step(model, X, y, criterion, lr):
    """
    Returns: loss value as a Python float
    """
    pred = model(X)
    loss = criterion(pred, y)

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.data.sub_(lr * param.grad)

                param.grad.zero_()

    return loss.item()