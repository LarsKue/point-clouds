import torch

from typing import Callable


def _except(f: Callable, x: torch.Tensor, *dim, **kwargs):
    """ Apply f on all dimensions except those specified in dim """
    result = x
    dimensions = [d for d in range(x.dim()) if d not in dim]

    if not dimensions:
        raise ValueError(f"Cannot exclude dims {dim} from x with shape {x.shape}: No dimensions left.")

    return f(result, dim=dimensions, **kwargs)


def sum_except(x: torch.Tensor, *dim):
    """ Sum all dimensions of x except the ones specified in dim """
    return _except(torch.sum, x, *dim)


def sum_except_batch(x):
    """ Sum all dimensions of x except the batch dimension """
    return sum_except(x, 0)


def mean_except(x: torch.Tensor, *dim):
    """ See sum_except """
    return _except(torch.mean, x, *dim)


def mean_except_batch(x):
    """ See sum_except_batch """
    return mean_except(x, 0)


def std_except(x: torch.Tensor, *dim):
    return _except(torch.std, x, *dim)


def std_except_batch(x):
    return std_except(x, 0)
