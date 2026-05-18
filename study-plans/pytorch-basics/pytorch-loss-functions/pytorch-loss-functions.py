import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    pred, target = torch.as_tensor(pred), torch.as_tensor(target)
    
    if method == 'mse':
        target = torch.as_tensor(target, dtype=torch.float32)
        mse = ((pred - target) ** 2).mean()
        return mse
        
    elif method == 'cross_entropy':
        pred = torch.as_tensor(pred, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.long)
        
        log_probabiility = torch.nn.LogSoftmax(dim=1)(pred)
        criterion = torch.nn.NLLLoss()
        
        cross_entropy = criterion(log_probabiility, target)
        return cross_entropy
        
    else:
        target = torch.as_tensor(target, dtype=torch.float32)
        
        error = torch.abs((pred - target))
        mse_part = (error ** 2) / 2
        mae_part = delta * (error - delta / 2)

        huber = torch.where(error <= delta, mse_part, mae_part)
        return huber.mean()