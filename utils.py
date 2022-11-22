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


def norm_except(x, *dim):
    return _except(torch.norm, x, *dim)


def norm_except_batch(x):
    return norm_except(x, 0)


def repeat_as(x1: torch.Tensor, x2: torch.Tensor):
    """ Repeat x1 to match the shape of x2 """
    if x1.dim() != x2.dim():
        raise RuntimeError(f"Tensors must have matching dimension.")

    s1 = torch.tensor(x1.shape)
    s2 = torch.tensor(x2.shape)

    div = s2 // s1
    mod = s2 % s1
    if torch.any(torch.nonzero(mod)):
        raise RuntimeError(f"Cannot repeat tensor of shape {x1.shape} to match {x2.shape}.")

    return x1.repeat(div.tolist())


def repeat_dim(x: torch.Tensor, count: int, *, dim: int):
    s = torch.ones(x.dim(), dtype=torch.int32)
    s[dim] = count

    return x.repeat(*s.tolist())


def unsqueeze_to(x: torch.Tensor, dim: int, side="right"):
    """ Unsqueeze x1 on the right to match the given dimensionality """
    if dim < x.dim():
        raise RuntimeError(f"Cannot unsqueeze tensor of dim {x.dim()} to {dim}.")

    idx = [None] * (dim - x.dim())
    if side == "right":
        idx = [..., *idx]
    elif side == "left":
        idx = [*idx, ...]
    else:
        raise ValueError(f"Unknown side: {side}")

    return x[idx]


def unsqueeze_as(x1: torch.Tensor, x2: torch.Tensor, **kwargs):
    return unsqueeze_to(x1, x2.dim(), **kwargs)


def normalize(x: torch.Tensor):
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)

    return (x - mean) / std
