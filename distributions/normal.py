
import torch

import utils

from .base import Distribution


class StandardNormal(Distribution):
    """
    Multivariate Distribution with zero mean and unit covariance
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        # we can internally use a univariate normal distribution since the covariance is diagonal
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def sample(self, shape=torch.Size()) -> torch.Tensor:
        shape = (*shape, *self.shape)
        return self.dist.sample(shape)

    def log_prob(self, x: torch.Tensor, *, condition: torch.Tensor = None) -> torch.Tensor:
        # the likelihood is the product of all individual likelihoods
        # as given by the probability density function (pdf)
        # since we take the log, we can take the sum of log pdfs instead
        return utils.sum_except_batch(self.dist.log_prob(x))
