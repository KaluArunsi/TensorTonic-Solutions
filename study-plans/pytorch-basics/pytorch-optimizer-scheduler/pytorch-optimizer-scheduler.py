import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    all_lrs = []
    avg_loss = []

    for i in range (1, num_epochs + 1):
        total_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        all_lrs.append(learning_rate)
        
        for input, target in dataloader:
            optimizer.zero_grad()
            
            pred = model(input)
            loss = criterion(pred, target)
            loss.backward()
                
            optimizer.step()

            total_loss += loss.item()

        avg_loss.append(total_loss / len(dataloader))
        scheduler.step()

    return {
        'losses': avg_loss,
        'lrs': all_lrs
    }