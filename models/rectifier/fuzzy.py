
import torch
from torch.utils.data import TensorDataset

from .base import Rectifier
from models.trainable import Trainable
from models.utils import fuzzy_match

import utils


class FuzzyRectifier(Rectifier, Trainable):

    @property
    def default_hparams(self):
        return super().default_hparams | dict(

        )

    def __init__(self, *datasets: TensorDataset, **hparams):
        super().__init__(*datasets, **hparams)

    def loss(self, batch, batch_idx):
        x0, mean, std = batch
        x1 = mean + std * torch.randn_like(x0)

        # TODO: condition

        samples = self.hparams.time_samples

        t = torch.rand(x0.shape[0] * samples).to(self.device)
        t = utils.unsqueeze_as(t, x0)

        x0 = utils.repeat_dim(x0, samples, dim=0)
        x1 = utils.repeat_dim(x1, samples, dim=0)

        xt = t * x1 + (1 - t) * x0

        v = self.velocity(xt, t)
        v_target = x1 - x0

        # use norm instead of mse for increased flexibility
        return torch.linalg.norm(v - v_target, ord=self.hparams.norm, dim=-1).mean()

    def rectify(self, condition: torch.Tensor = None, steps: int = 100, samples: int = None):
        x = self.train_data.tensors[0].to(self.device)

        simulated, _ = self.forward(x, condition=condition, steps=steps)
        sampled = self.distribution.sample(simulated.shape[:-1]).to(self.device)

        # find means and stds of fuzzy match
        means, stds = fuzzy_match(sampled, simulated)

        self.train_data.tensors = [x, means, stds]
