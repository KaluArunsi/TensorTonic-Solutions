import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    all_losses = []
    
    for input, target in dataloader:
        optimizer.zero_grad()
        forward_pass = model(input)
        
        loss_function = criterion(forward_pass, target)
        all_losses.append(loss_function)
        loss_function.backward()
        
        optimizer.step()

    return sum(all_losses) / len(all_losses)