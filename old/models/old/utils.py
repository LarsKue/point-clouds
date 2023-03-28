
import torch.nn as nn

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

import warnings
from point_clouds.models.encoder.pools.self_attention import GlobalMultimaxPool1d


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


def find_closest(x: jnp.array, ys: jnp.array) -> (jnp.array, jnp.array):
    """ Find the closest y to x """
    residuals = x[None] - ys
    norms = jnp.linalg.norm(residuals, axis=-1)

    i = jnp.argmin(norms, axis=0)

    return ys[i], norms[i]

find_closest = jit(vmap(find_closest, in_axes=(0, None)))


def fuzzy_match(xs: np.array, ys: np.array, batch_size: int = None) -> (np.array, np.array):
    """ Find means and standard deviations for a fuzzy matching of y in x """
    xs = xs.reshape(-1, xs.shape[-1])
    ys = ys.reshape(-1, ys.shape[-1])

    if batch_size is None:
        means, stds = find_closest(ys, xs)
    else:
        means = []
        stds = []
        for start in range(0, len(ys), batch_size):
            y_batch = ys[start:start + batch_size]
            mean_batch, std_batch = find_closest(y_batch, xs)
            means.append(mean_batch)
            stds.append(std_batch)

        means = np.concatenate(means, axis=0)
        stds = np.concatenate(stds, axis=0)

    return np.asarray(means), np.asarray(stds)


