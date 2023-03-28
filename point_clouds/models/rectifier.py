
import torch

from lightning_trainable.modules import DenseModule, DenseModuleHParams

from point_clouds import integrators


class RectifierHParams(DenseModuleHParams):
    conditions: int
    integrator: str = "euler"

    @classmethod
    def validate_parameters(cls, hparams: dict) -> dict:
        hparams["outputs"] = hparams["inputs"]
        hparams["inputs"] = hparams["inputs"] + 1 + hparams["conditions"]

        hparams = super().validate_parameters(hparams)

        return hparams


class Rectifier(DenseModule):
    hparams: RectifierHParams

    def __init__(self, hparams: RectifierHParams | dict):
        super().__init__(hparams)
        self.integrator = self.configure_integrator()

    def velocity(self, batch: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size, points, dims = batch.shape
        assert time.shape == (batch_size, 1, 1)
        conditions = self.hparams.conditions
        assert condition.shape == (batch_size, 1, conditions)


        x = torch.cat((batch, time, condition), dim=2)
        return self.network(x)

    def inverse(self, noise: torch.Tensor, condition: torch.Tensor, steps: int = 100):
        batch_size, points, dims = noise.shape

        time = torch.ones((batch_size, 1, 1))
        time_step = -torch.ones_like(time) / steps

        def f(x, t):
            return self.velocity(x, t, condition)

        batch, _ = self.integrator.solve(f=f, x0=noise, t0=time, dt=time_step, steps=steps)

        return batch

    def configure_integrator(self):
        match self.hparams.integrator.lower():
            case "euler":
                return integrators.EulerIntegrator()
            case "rk45":
                return integrators.RK45Integrator()
            case other:
                raise NotImplementedError(f"Unrecognized integrator: {other}")
