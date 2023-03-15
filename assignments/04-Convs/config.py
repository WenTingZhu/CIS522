from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    # Resize,
    # RandomCrop,
    # RandomHorizontalFlip,
)


class CONFIG:
    batch_size = 200
    num_epochs = 8

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=6e-3)

    transforms = Compose(
        [
            # RandomCrop(
            #     16, padding=4
            # ),  # randomly crop the image to 32x32 with padding of 4 pixels
            # RandomHorizontalFlip(),  # randomly flip the image horizontally
            # Resize(28),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
