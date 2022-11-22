
import torch
import torch.nn as nn

from typing import Tuple

from integrators import Integrator

import utils


class NeuralODE(nn.Module):
    def __init__(self, network: nn.Module, integrator: Integrator):
        super().__init__()
        self.network = network
        self.integrator = integrator

    def forward(self, x: torch.Tensor, condition: torch.Tensor = torch.Tensor(), steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        def v(x, t):
            return self.velocity(x, t, condition=condition)

        t0 = torch.zeros(x.shape[0])
        t0 = utils.unsqueeze_as(t0, x)
        dt = torch.tensor(1.0 / steps)

        return self.integrator.solve(v, x0=x, t0=t0, dt=dt, steps=steps)

    def inverse(self, z: torch.Tensor, condition: torch.Tensor = torch.Tensor(), steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        def v(x, t):
            return self.velocity(x, t, condition=condition)

        t0 = torch.ones(z.shape[0])
        t0 = utils.unsqueeze_as(t0, z)
        dt = torch.tensor(-1.0 / steps)

        return self.integrator.solve(v, x0=z, t0=t0, dt=dt, steps=steps)

    def velocity(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor = torch.Tensor()) -> torch.Tensor:
        # x: (shapes, points, dim)
        # t: (shapes, 1, 1)
        # c: (shapes, condition_dim)
        n_shapes, n_points, n_dim = x.shape

        x = torch.flatten(x, 0, 1)
        time = utils.repeat_dim(time, n_points, dim=0).squeeze(-1)
        condition = utils.repeat_dim(condition, n_points, dim=0)

        # (shapes * points, dim + 1 + condition_dim)
        xtc = torch.cat((x, time, condition), dim=1)

        velocity = self.network.forward(xtc)
        velocity = torch.movedim(velocity, -1, 1)
        velocity = torch.unflatten(velocity, dim=0, sizes=(n_shapes, n_points))

        return velocity
