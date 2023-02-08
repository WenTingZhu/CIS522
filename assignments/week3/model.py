import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.activation = activation
        self.init = initializer

        # Initialize layers of MLP
        self.layers = nn.ModuleList()

        for i in range(hidden_count):
            self.layers += [nn.Linear(input_size, hidden_size)]
            # self.layers += [nn.BatchNorm1d(hidden_size)]
            self.layers += [self.activation]
            self.layers += [nn.Dropout(0.1)]
            input_size = hidden_size

        # Create final layer
        self.out = nn.Linear(hidden_size, num_classes)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init(layer.weight)
        self.init(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # Flatten inputs to 2D (if more than that)
        # x = x.flatten(start_dim=1)

        # Get activations of each layer
        for layer in self.layers:
            # x = self.activation(layer(x))
            x = layer(x)

        # Get outputs
        x = self.out(x)

        return x
