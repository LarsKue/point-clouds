
import torch


class Distribution:
    """
    Base class for primitive distributions as used by normalizing flows.
    These distributions have no associated tensors, thus saving memory,
    but sacrificing some utility functions

    Use these when working with large inputs, e.g. images.
    Otherwise, use torch.distributions.Distribution
    """
    def sample(self, shape=torch.Size()) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError
