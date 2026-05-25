import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    output_tensor = torch.empty(fan_out, fan_in)
    method_check = [
        'xavier_uniform',
        'xavier_normal',
        'he_uniform',
        'he_normal'
    ]

    if method not in method_check:
        raise ValueError("Method name incorrect. Must be xavier_uniform, xavier_normal, he_uniform, or he_normal.")

    if method == 'xavier_uniform':
        limit = torch.sqrt(torch.tensor(6 / (fan_in + fan_out)))
        return output_tensor.uniform_(-limit, limit)
        
    elif method == 'xavier_normal':
        std = torch.sqrt(torch.tensor(2 / (fan_in + fan_out)))
        return output_tensor.normal_(mean=0.0, std=std)
        
    elif method == 'he_uniform':
        limit = torch.sqrt(torch.tensor(6 / fan_in))
        return output_tensor.uniform_(-limit, limit)
        
    elif method == 'he_normal':
        std = torch.sqrt(torch.tensor(2 / fan_in))
        return output_tensor.normal_(mean=0.0, std=std)