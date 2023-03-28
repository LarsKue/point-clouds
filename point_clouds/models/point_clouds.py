
import torch
import torch.nn.functional as F
from lightning_trainable import Trainable, TrainableHParams

from .encoder import InvariantEncoder, InvariantEncoderHParams
from .rectifier import Rectifier, RectifierHParams


class PointCloudsModelHParams(TrainableHParams):
    inputs: int
    points: int
    conditions: int

    encoder_hparams: dict
    rectifier_hparams: dict


class PointCloudsModel(Trainable):
    hparams: PointCloudsModelHParams

    def __init__(self, hparams: PointCloudsModelHParams | dict, **kwargs):
        if not isinstance(hparams, PointCloudsModelHParams):
            hparams = PointCloudsModelHParams(**hparams)

        super().__init__(hparams, **kwargs)
        self.encoder = self.configure_encoder()
        self.rectifier = self.configure_rectifier()
        self.integrator = self.configure_integrator()

    def compute_metrics(self, batch, batch_idx) -> dict:
        if not torch.is_tensor(batch):
            batch = batch[0]
            assert torch.is_tensor(batch)

        batch_size, points, dims = batch.shape

        time = torch.rand((batch_size, 1, 1))
        condition = self.encoder(batch)
        predicted_velocity = self.rectifier.velocity(batch, time, condition)

        noise = torch.randn_like(batch)
        target_velocity = noise - batch

        mse = F.mse_loss(predicted_velocity, target_velocity)

        return dict(
            mse=mse
        )

    def sample(self, sample_shape=(1, 2048), steps: int = 100) -> torch.Tensor:
        (batch_size, points) = sample_shape

        noise = torch.randn(batch_size, points, self.hparams.inputs)
        condition = torch.randn(batch_size, 1, self.hparams.conditions)

        return self.rectifier.inverse(noise, condition, steps)

    def configure_encoder(self):
        module = self.hparams.encoder_hparams["equivariant_module"]
        test_batch = torch.randn(1, self.hparams.points, self.hparams.inputs, device=self.device)
        test_batch = module(test_batch)
        hparams = dict(
            inputs=test_batch.shape[2],
            outputs=self.hparams.conditions,
            **self.hparams.encoder_hparams,
        )
        return InvariantEncoder(hparams)

    def configure_rectifier(self):
        hparams = dict(
            inputs=self.hparams.inputs,
            conditions=self.hparams.conditions,
            **self.hparams.rectifier_hparams,
        )
        return Rectifier(hparams)
