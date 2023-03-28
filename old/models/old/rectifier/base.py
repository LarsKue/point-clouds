
import torch
import torch.nn as nn
import torch.distributions as D
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

import pytorch_lightning as lightning

from typing import Tuple

from old.models.old.utils import make_dense
from old import utils, integrators


class Rectifier(lightning.LightningModule):

    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            inputs=0,
            conditions=0,
            time_samples=1,
            dropout=None,
            widths=[],
            activation="relu",
            checkpoints=None,
            integrator="euler",
            distribution="normal",
            norm=2,
        )

    def __init__(self, *datasets, **hparams):
        super().__init__(*datasets, **hparams)

        self.network = self.configure_network()
        self.distribution = self.configure_distribution()
        self.integrator = self.configure_integrator()

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None, steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def inverse(self, z: torch.Tensor, condition: torch.Tensor = None, steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def velocity(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the velocity field at given point and time for a given condition
        Parameters
        ----------
        x: Tensor of shape (batch_size, inputs) or (batch_size, sequence_size, inputs)
        time: Tensor of shape (batch_size,)
        condition: Tensor of shape (batch_size, conditions)

        Returns
        -------
        velocity: Tensor of shape (batch_size, inputs) or (batch_size, sequence_size, inputs)
        """
        time = utils.unsqueeze_as(time, x)

        if x.dim() == 3:
            # repeat condition and time for sequence
            sequence_size = x.shape[1]
            time = utils.repeat_dim(time, sequence_size, dim=1)
            if condition:
                condition = utils.repeat_dim(condition.unsqueeze(1), sequence_size, dim=1)

        if condition:
            xtc = torch.cat((x, time, condition), dim=-1)
        else:
            xtc = torch.cat((x, time), dim=-1)

        if not self.training or not self.hparams.checkpoints:
            velocity = self.network(xtc)
        else:
            if self.hparams.checkpoints is True:
                velocity = checkpoint(
                    self.network,
                    xtc,
                    use_reentrant=False
                )
            else:
                velocity = checkpoint_sequential(
                    functions=self.network,
                    segments=self.hparams.checkpoints,
                    input=xtc
                )

        return velocity

    def configure_network(self) -> nn.Module:
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
            dropout=self.hparams.dropout
        )

        return network

    def configure_distribution(self) -> D.Distribution:
        inputs = self.hparams.inputs

        match self.hparams.distribution.lower():
            case "normal" | "gaussian":
                return D.Normal(torch.zeros(inputs), torch.ones(inputs))
            case "student-t":
                return D.StudentT(inputs - 1, torch.zeros(inputs), torch.ones(inputs))
            case "uniform":
                return D.Uniform(-torch.ones(inputs), torch.ones(inputs))
            case distribution:
                raise NotImplementedError(f"Unknown distribution: {distribution}")

    def configure_integrator(self) -> integrators.Integrator:
        """ Configure and return integrator used for inference """
        match self.hparams.integrator.lower():
            case "euler":
                integrator = integrators.EulerIntegrator()
            case "rk45":
                integrator = integrators.RK45Integrator()
            case integrator:
                raise NotImplementedError(f"Unsupported Integrator: {integrator}")

        return integrator
