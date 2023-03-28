
from .integrator import Integrator


class RK45Integrator(Integrator):
    def step(self, f, x, t, dt):
        k1 = f(x, t)
        k2 = f(x + dt * 0.5 * k1, t + 0.5 * dt)
        k3 = f(x + dt * 0.5 * k2, t + 0.5 * dt)
        k4 = f(x + dt * k3, t + dt)

        return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0, t + dt
