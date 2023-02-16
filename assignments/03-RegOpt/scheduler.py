from typing import List
from torch.optim.lr_scheduler import _LRScheduler
import torch
import math


class CustomLRScheduler(_LRScheduler):
    """
    A custom defined learning rate sheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List = None,
        step_size: int = None,
        gamma: int = None,
        start_epoch: int = None,
        T_0: int = None,
        T_mult: int = None,
        eta_min: int = None,
        lr_lambda: list = None,
        last_epoch: int = -1,
    ) -> None:
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """

        if lr_lambda is not None:
            self.lr_lambdas = lr_lambda
        if step_size is not None:
            self.step_size = step_size
        if gamma is not None:
            self.gamma = gamma
        if start_epoch is not None:
            self.start_epoch = start_epoch
        if milestones is not None:
            self.milestones = milestones
        if T_0 is not None:
            self.T_0 = T_0
            self.T_i = T_0
        if T_mult is not None:
            self.T_mult = T_mult
        if eta_min is not None:
            self.eta_min = eta_min

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return learning rate

        Returns:
            List[float]: learning rate
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # return [i for i in self.base_lrs]
        # LambdaLR
        # return [base_lr * lmbda(self.last_epoch)
        #         for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        # MultiplicativeLR
        # if self.last_epoch > self.start_epoch:
        #     return [group['lr'] * lmbda(self.last_epoch)
        #             for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)]
        # else:
        #     return [group['lr'] for group in self.optimizer.param_groups]
        # StepLR
        # if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
        #     return [group['lr'] for group in self.optimizer.param_groups]
        # return [group['lr'] * self.gamma
        #         for group in self.optimizer.param_groups]
        # MultiStepLR
        # if self.last_epoch not in self.milestones:
        #     return [group["lr"] for group in self.optimizer.param_groups]
        # return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        # Cosine
        self.T_i *= self.T_mult ** (self.last_epoch // self.T_i)
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch % self.T_i) / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    # def _get_closed_form_lr(self) -> List[float]:
    #     """
    #     Return closed form solution for learning rate

    #     Returns:
    #         List[float]: learning rate
    #     """
    #     return [
    #         base_lr * self.gamma ** (self.last_epoch // self.step_size)
    #         for base_lr in self.base_lrs
    #     ]
