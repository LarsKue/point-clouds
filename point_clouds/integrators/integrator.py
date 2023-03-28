
from typing import Callable, Protocol, Tuple

from torch import Tensor


Integratable = Callable[[Tensor, Tensor], Tensor]


class Integrator(Protocol):
    def step(self, f: Integratable, x: Tensor, t: Tensor, dt: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def solve(self, f: Integratable, x0: Tensor, t0: Tensor, dt: Tensor, steps: int) -> Tuple[Tensor, Tensor]:
        x = x0
        t = t0
        for step in range(steps):
            x, t = self.step(f, x, t, dt)

        return x, t
