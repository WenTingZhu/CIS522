from typing import List
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
import torch

# import math
import weakref


class CustomLRScheduler(_LRScheduler):
    """
    A custom defined learning rate sheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List = None,
        step_size: int = None,
        gamma: float = None,
        start_epoch: int = None,
        T_0: int = None,
        T_mult: int = None,
        eta_min: float = None,
        lr_lambda: list = None,
        last_epoch: int = -1,
        base_lr: float = None,
        max_lr: float = None,
        step_size_up: int = 2000,
        step_size_down: int = None,
        mode: str = "triangular",
        scale_fn: str = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = False,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
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

        self.eta_min = eta_min
        self.milestones = milestones

        self.optimizer = optimizer

        base_lrs = self._format_param("base_lr", optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group["lr"] = lr

        self.max_lrs = self._format_param("max_lr", optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = (
            float(step_size_down) if step_size_down is not None else step_size_up
        )
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            self._scale_fn_custom = None
            if self.mode == "triangular":
                self._scale_fn_ref = weakref.WeakMethod(self._triangular_scale_fn)
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self._scale_fn_ref = weakref.WeakMethod(self._triangular2_scale_fn)
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self._scale_fn_ref = weakref.WeakMethod(self._exp_range_scale_fn)
                self.scale_mode = "iterations"
        else:
            self._scale_fn_custom = scale_fn
            self._scale_fn_ref = None
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer.defaults:
                raise ValueError(
                    "optimizer must support momentum with `cycle_momentum` option enabled"
                )

            base_momentums = self._format_param(
                "base_momentum", optimizer, base_momentum
            )
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group["momentum"] = momentum
            self.base_momentums = [
                group["momentum"] for group in optimizer.param_groups
            ]
            self.max_momentums = self._format_param(
                "max_momentum", optimizer, max_momentum
            )

        self.base_lrs = base_lrs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} values for {}, got {}".format(
                        len(optimizer.param_groups), name, len(param)
                    )
                )
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def scale_fn(self, x):
        """Return scale function"""
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)

        else:
            return self._scale_fn_ref()(x)

    def _triangular_scale_fn(self, x):
        """Return triangle scale function"""
        return 1.0

    def _triangular2_scale_fn(self, x):
        """Return triangle scale function"""
        return 1 / (2.0 ** (x - 1))

    def _exp_range_scale_fn(self, x):
        """Return exponential scale function"""
        return self.gamma ** (x)

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
        # self.T_i *= self.T_mult ** (self.last_epoch // self.T_i)
        # return [
        #     self.eta_min
        #     + (base_lr * self.gamma ** (self.last_epoch // self.T_i) - self.eta_min)
        #     * (1 + math.cos(math.pi * (self.last_epoch % self.T_i) / self.T_i))
        #     / 2
        #     for base_lr in self.base_lrs
        # ]
        # CyclicLR
        # cycle = math.floor(1 + self.last_epoch / self.total_size)
        # x = 1. + self.last_epoch / self.total_size - cycle
        # if x <= self.step_ratio:
        #     scale_factor = x / self.step_ratio
        # else:
        #     scale_factor = (x - 1) / (self.step_ratio - 1)

        # lrs = []
        # for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
        #     base_height = (max_lr - base_lr) * scale_factor
        #     if self.scale_mode == 'cycle':
        #         lr = base_lr + base_height * self.scale_fn(cycle)
        #     else:
        #         lr = base_lr + base_height * self.scale_fn(self.last_epoch)
        #     lrs.append(lr)

        # if self.cycle_momentum:
        #     momentums = []
        #     for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
        #         base_height = (max_momentum - base_momentum) * scale_factor
        #         if self.scale_mode == 'cycle':
        #             momentum = max_momentum - base_height * self.scale_fn(cycle)
        #         else:
        #             momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
        #         momentums.append(momentum)
        #     for param_group, momentum in zip(self.optimizer.param_groups, momentums):
        #         param_group['momentum'] = momentum

        # return lrs
        # CyclicLinearLR
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx - 1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 - 1.0 * curr_pos / width)
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
