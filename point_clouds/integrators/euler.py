
from .integrator import Integrator


class EulerIntegrator(Integrator):
    def step(self, f, x, t, dt):
        return x + f(x, t) * dt, t + dt
