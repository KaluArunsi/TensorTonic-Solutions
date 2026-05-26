import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    Returns: loss or None from step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum

        if self.lr < 0.0:
            raise ValueError(f"Invalid Learning Rate here: {self.lr}\n Learning Rate can't be less than zero.")

        if self.momentum < 0.0:
            raise ValueError(f"Invalid Momentum here: {self.momentum}\nMomentum can't be less than zero.")

        defaults = {'lr': self.lr, 'momentum': self.momentum}

        super().__init__(self.params, defaults)

    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self.lr = group['lr']
            self.momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
    
                if len(state) == 0.0:
                    state['velocity'] = torch.zeros_like(p.data)
        
                v = state['velocity']
        
                v.mul_(self.momentum).add_(grad)
                p.data.add_(v, alpha=-self.lr)

        return loss