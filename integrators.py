
import torch

from typing import Callable


class Integrator:
    def step(self, f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dt: float) -> torch.Tensor:
        raise NotImplementedError

    def solve(self, f: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, dt: float, steps: int):
        x = x0
        for step in range(steps):
            x = self.step(f, x, dt)

        return x


class EulerIntegrator(Integrator):
    def step(self, f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dt: float) -> torch.Tensor:
        return x + f(x) * dt


class RK45Integrator(Integrator):
    def step(self, f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = f(x)
        k2 = f(x + dt * 0.5 * k1)
        k3 = f(x + dt * 0.5 * k2)
        k4 = f(x + dt * k3)

        return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
