
import torch

from typing import Callable, Tuple


class Integrator:
    def step(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def solve(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x0: torch.Tensor, t0: torch.Tensor, dt: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x0
        t = t0
        for step in range(steps):
            x, t = self.step(f, x, t, dt)

        return x, t


class EulerIntegrator(Integrator):
    def step(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + f(x, t) * dt, t + dt


class RK45Integrator(Integrator):
    def step(self, f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k1 = f(x, t)
        k2 = f(x + dt * 0.5 * k1, t + 0.5 * dt)
        k3 = f(x + dt * 0.5 * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)

        return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0, t + dt
