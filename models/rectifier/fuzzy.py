
import torch
from torch.utils.data import TensorDataset

from .base import Rectifier
from models.trainable import Trainable

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

        # simulated latent data
        simulated, _ = self.forward(x, condition=condition, steps=steps)

        # match simulated to sampled data
        # TODO: more samples according to `samples` parameter
        sampled = self.distribution.sample(simulated.shape[:-1]).to(self.device)

        simulated = torch.flatten(simulated, 0, -2)
        sampled = torch.flatten(sampled, 0, -2)

        # fuzzy match: use nearest sample to recompute
        residuals = simulated[None, :] - sampled[:, None]
        norms = torch.linalg.norm(residuals, ord=self.hparams.norm, dim=-1)

        values, indices = torch.topk(-norms.ravel(), k=torch.prod(torch.tensor(sampled.shape[:-1], dtype=torch.int64)).item())
        values = -values

        means = sampled[indices].reshape_as(x)
        stds = values.reshape_as(x)

        self.train_data.tensors = [x, means, stds]
