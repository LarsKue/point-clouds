
import pytorch_lightning as lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from integrators import EulerIntegrator, RK45Integrator
from .utils import make_dense

import utils


class Rectifier(lightning.LightningModule):

    @property
    def default_hparams(self):
        return dict(
            inputs=0,
            conditions=0,
            dropout=None,
            batchnorm=False,
            widths=[],
            activation="relu",
            integrator="euler",
        )

    def __init__(self, **hparams):
        super().__init__()
        # TODO: check if these need to be saved here or in wrapping class
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.network = self.configure_network()
        self.integrator = self.configure_integrator()
        self.distribution = self.configure_distribution()

    def forward(self, x: torch.Tensor, condition: torch.Tensor = torch.Tensor(), steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a full forward integration for given data point x and shape condition
        Parameters
        ----------
        x: Tensor of shape (shapes, points, dim)
        condition: Tensor of shape (shapes, conditions)
        steps: Number of integration steps to use. Higher values yield more accurate results, but are slower to compute.

        Returns
        -------
        z: Latent Tensor of shape (shapes, points, dim)
        """
        def v(x, t):
            return self.velocity(x, t, condition=condition)

        t0 = torch.zeros(x.shape[0])
        t0 = utils.unsqueeze_as(t0, x)
        dt = torch.tensor(1.0 / steps)

        return self.integrator.solve(v, x0=x, t0=t0, dt=dt, steps=steps)

    def inverse(self, z: torch.Tensor, condition: torch.Tensor = torch.Tensor(), steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a full inverse integration for given latent point z and shape condition
        Parameters
        ----------
        z: Tensor of shape (shapes, points, dim)
        condition: Tensor of shape (shapes, conditions)
        steps: Number of integration steps to use. Higher values yield more accurate results, but are slower to compute.

        Returns
        -------
        x: Tensor of shape (shapes, points, dim)
        """
        def v(x, t):
            return self.velocity(x, t, condition=condition)

        t0 = torch.ones(z.shape[0])
        t0 = utils.unsqueeze_as(t0, z)
        dt = torch.tensor(-1.0 / steps)

        return self.integrator.solve(v, x0=z, t0=t0, dt=dt, steps=steps)

    def velocity(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor = torch.Tensor()) -> torch.Tensor:
        """
        Compute the velocity field at given point and time for a given condition
        Parameters
        ----------
        x: Tensor of shape (shapes, points, dim)
        time: Tensor of shape (shapes, 1, 1)
        condition: Tensor of shape (shapes, conditions)

        Returns
        -------
        velocity: Tensor of shape (shapes, points, dim)
        """
        n_shapes, n_points, n_dim = x.shape

        # (shapes, points, 1)
        time = utils.repeat_dim(time, n_points, dim=1)
        # (shapes, points, condition_dim)
        condition = utils.repeat_dim(condition.unsqueeze(1), n_points, dim=1)

        # (shapes, points, dim + 1 + condition_dim)
        xtc = torch.cat((x, time, condition), dim=-1)

        # velocity.shape == (shapes, points, dim)
        velocity = self.network.forward(xtc)

        # velocity.shape == (shapes, points, dim)
        return velocity

    def loss(self, points: torch.Tensor, condition: torch.Tensor, samples: int):
        """
        Compute the loss for the rectifier
        Parameters
        ----------
        points: The input points to the rectifier. Tensor of shape (shapes, points, dim)
        condition: The input condition. Tensor of shape (shapes, conditions)
        samples: How many time samples to use.

        Returns
        -------
        loss: Tensor of shape ()
        """
        n_shapes, n_points, dim = points.shape

        target_noise = self.distribution.sample((n_shapes, n_points)).to(self.device)

        # (samples * shapes, 1, 1)
        time = torch.rand(samples * n_shapes).to(self.device)
        time = utils.unsqueeze_as(time, points)

        target_noise = utils.repeat_dim(target_noise, samples, dim=0)
        points = utils.repeat_dim(points, samples, dim=0)
        condition = utils.repeat_dim(condition, samples, dim=0)

        # (samples * shapes, points, dim)
        interpolation = time * target_noise + (1 - time) * points

        # (samples * shapes, points, dim)
        v = self.velocity(interpolation, time=time, condition=condition)

        # (samples * shapes, points, dim)
        v_target = target_noise - points

        return F.mse_loss(v, v_target)

    def configure_network(self):
        """
        Configure and Return the Network to use based on hparams
        Returns
        -------
        network: nn.Module
        """
        inputs = self.hparams.inputs
        conditions = self.hparams.conditions
        widths = self.hparams.widths

        network = make_dense(
            widths=[inputs + 1 + conditions, *widths, inputs],
            activation=self.hparams.activation,
            batchnorm=self.hparams.batchnorm,
            dropout=self.hparams.dropout
        )

        return network

    def configure_integrator(self):
        """ Configure and return integrator used for inference """
        match self.hparams.integrator.lower():
            case "euler":
                integrator = EulerIntegrator()
            case "rk45":
                integrator = RK45Integrator()
            case integrator:
                raise NotImplementedError(f"Unsupported Integrator: {integrator}")

        return integrator

    def configure_distribution(self):
        loc = torch.zeros(self.hparams.inputs).cuda()
        scale = torch.ones(self.hparams.inputs).cuda()
        return torch.distributions.Normal(loc, scale)
