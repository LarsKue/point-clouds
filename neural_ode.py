
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

        t0 = torch.zeros_like(x)
        dt = torch.tensor(1.0 / steps)

        return self.integrator.solve(v, x0=x, t0=t0, dt=dt, steps=steps)

    def inverse(self, z: torch.Tensor, condition: torch.Tensor = torch.Tensor(), steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        def v(x, t):
            return self.velocity(x, t, condition=condition)

        t0 = torch.ones_like(z)
        dt = torch.tensor(-1.0 / steps)

        return self.integrator.solve(v, x0=z, t0=t0, dt=dt, steps=steps)

    def velocity(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor = torch.Tensor()) -> torch.Tensor:
        time = utils.unsqueeze_as(time, x)
        condition = utils.unsqueeze_as(condition, x)
        xtc = torch.cat((x, time, condition), dim=1)

        return self.network.forward(xtc)
