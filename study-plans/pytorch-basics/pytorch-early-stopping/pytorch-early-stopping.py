import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    counter = 0
    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            pred = model(input_batch)
            loss = criterion(pred, target_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                pred = model(input_batch)
                loss = criterion(pred, target_batch)
                total_val_loss += loss.item()

        current_val_loss = total_val_loss / len(val_loader)
        val_losses.append(current_val_loss)

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            counter = 0
        else:
            counter += 1
    
        if counter >= patience:
            stopped_epoch = epoch
            break

    return {'train_losses': train_losses,
            'val_losses': val_losses,
            'stopped_epoch': stopped_epoch
           }