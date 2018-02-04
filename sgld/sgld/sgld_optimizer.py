import torch
from torch.distributions import Normal
from torch.optim.optimizer import Optimizer, required
import numpy as np



class SGLD(Optimizer):
    """
    Barely modified version of pytorch SGD to implement SGLD
    """

    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)

    def step(self, lr, add_noise = False):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if add_noise:
                    size = d_p.size()
                    langevin_noise = Normal(
                        torch.zeros(size),
                        torch.ones(size) * np.sqrt(lr)
                    )
                    p.data.add_(-group['lr'],
                                d_p + langevin_noise.sample().cuda())
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss