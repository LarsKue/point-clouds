
import torch.nn as nn

import numpy as np

import warnings
from .pools import GlobalMultimaxPool1d


def get_activation(activation):
    match activation.lower():
        case "relu":
            return nn.ReLU
        case "elu":
            return nn.ELU
        case "selu":
            return nn.SELU
        case activation:
            raise NotImplementedError(f"Unsupported Activation: {activation}")


def get_pool(pool):
    match pool.lower():
        case "multimax":
            return GlobalMultimaxPool1d
        case pool:
            raise NotImplementedError(f"Unsupported Pool: {pool}")


def make_conv(widths: list[int], activation: str, dropout: float = None):
    if len(widths) < 2:
        raise ValueError(f"Need at least Input and Output Layer.")
    elif len(widths) < 3:
        warnings.warn(f"Should use more than zero hidden layers.")

    Activation = get_activation(activation)

    network = nn.Sequential()

    input_layer = nn.Conv1d(in_channels=widths[0], out_channels=widths[1], kernel_size=1)
    network.add_module("Input Layer", input_layer)
    network.add_module("Input Activation", Activation())

    for i in range(1, len(widths) - 2):
        if dropout is not None:
            network.add_module(f"Dropout {i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Conv1d(in_channels=widths[i], out_channels=widths[i + 1], kernel_size=1)
        network.add_module(f"Hidden Layer {i}", hidden_layer)
        network.add_module(f"Hidden Activation {i}", Activation())

    output_layer = nn.Conv1d(in_channels=widths[-2], out_channels=widths[-1], kernel_size=1)
    network.add_module("Output Layer", output_layer)

    return network


def make_dense(widths: list[int], activation: str, dropout: float = None):
    if len(widths) < 2:
        raise ValueError(f"Need at least Input and Output Layer.")
    elif len(widths) < 3:
        warnings.warn(f"Should use more than zero hidden layers.")

    Activation = get_activation(activation)

    network = nn.Sequential()

    # input is x, time, condition
    input_layer = nn.Linear(in_features=widths[0], out_features=widths[1])
    network.add_module("Input Layer", input_layer)
    network.add_module("Input Activation", Activation())

    for i in range(1, len(widths) - 2):
        if dropout is not None:
            network.add_module(f"Dropout {i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden Layer {i}", hidden_layer)
        network.add_module(f"Hidden Activation {i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output Layer", output_layer)

    return network
