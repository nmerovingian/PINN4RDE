import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    def __init__(self, units, activation='tanh', kernel_initializer='he_normal'):
        super(ResidualUnit, self).__init__()
        # Initialize the activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu  # Placeholder for other activations

        # Main layers
        self.main_layers = nn.Sequential(
            nn.Linear(units, units),
            nn.Linear(units, units)
        )
        self.initialize_weights(kernel_initializer)

        # Skip layers (if any additional layers were to be added)
        self.skip_layers = nn.Sequential(
            # Example: nn.Linear(units, units)
        )

    def forward(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = self.activation(layer(Z))

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = self.activation(layer(skip_Z))

        return self.activation(Z + skip_Z)

    def initialize_weights(self, kernel_initializer):
        if kernel_initializer == 'he_normal':
            init_fn = nn.init.kaiming_normal_
        else:
            init_fn = nn.init.xavier_normal_  # Placeholder for other initializers

        for layer in self.main_layers:
            if isinstance(layer, nn.Linear):
                init_fn(layer.weight)


class Network(nn.Module):
    def __init__(self, num_inputs=3, layers=[128, 64, 64, 64, 128], activation='tanh', num_outputs=1):
        super(Network, self).__init__()
        modules = []
        previous_units = num_inputs

        for units in layers:
            modules.append(nn.Linear(previous_units, units))
            if activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                modules.append(nn.ReLU())  # Placeholder for other activations
            previous_units = units

        modules.append(nn.Linear(layers[-1], num_outputs))
        self.layers = nn.Sequential(*modules)

        self.initialize_weights()

    def forward(self, x):
        return self.layers(x)

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

