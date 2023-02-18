from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    # RandomHorizontalFlip,
    # RandomCrop,
)


class CONFIG:
    """_summary_"""

    batch_size = 32
    num_epochs = 12
    initial_learning_rate = 0.0018
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        # 'lr_lambda': [lambda epoch: 0.999],
        # 'start_epoch': 20,
        # 'step_size': 100,
        "gamma": 0.95,
        # "milestones": [3000, 3600, 4000, 4800, 5400, 6000],
        "T_0": 32,
        "eta_min": 0.00035,
        "T_mult": 32,
    }  # gamma=0.9, milestones=[30,80]

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
        amsgrad=True,
    )  # lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False

    transforms = Compose(
        [
            # RandomHorizontalFlip(),
            # RandomCrop(32, padding=4),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
