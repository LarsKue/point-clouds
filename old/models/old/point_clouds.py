
import torch

from ... import utils

from .encoder import Encoder
# from .rectifier import Rectifier
from .trainable import Trainable


class PointCloudsModule(Trainable):

    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            augment_noise=0.05,
            time_samples=1,
            mmd_samples=None,
            mmd_scales="all",
            encoder_weight=1.0,
            rectifier_weight=1.0,
            encoder_hparams=dict(),
            rectifier_hparams=dict(),
        )

    def __init__(self, *datasets, **hparams):
        super().__init__(*datasets, **hparams)

        self.encoder = Encoder(**self.hparams.encoder_hparams)
        self.hparams.encoder_hparams = dict(self.encoder.hparams)
        self.rectifier = Rectifier(**self.hparams.rectifier_hparams)
        self.hparams.rectifier_hparams = dict(self.rectifier.hparams)

        self.save_hyperparameters(self.hparams)

    def inference_step(self, batch, batch_idx=None):
        points = utils.normalize(batch, dim=1)
        points = utils.augment(points, noise=self.hparams.augment_noise)

        n_shapes, n_points, dim = points.shape
        n_times = self.hparams.time_samples
        n_conditions = self.hparams.encoder_hparams["conditions"]

        condition = self.encoder.forward(points)
        assert condition.shape == (n_shapes, n_conditions)

        mmd_samples = self.hparams.mmd_samples or n_shapes * n_times

        encoder_loss = self.encoder.loss(condition, samples=mmd_samples, scales=self.hparams.mmd_scales)
        rectifier_loss = self.rectifier.loss(points, condition, samples=n_times)

        return encoder_loss, rectifier_loss

    def training_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)
        self.log("encoder_loss", encoder_loss.mean(dim=0))
        self.log("rectifier_loss", rectifier_loss.mean(dim=0))

        loss = self.hparams.encoder_weight * encoder_loss + self.hparams.rectifier_weight * rectifier_loss
        loss = loss.mean(dim=0)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)

        loss = self.hparams.encoder_weight * encoder_loss + self.hparams.rectifier_weight * rectifier_loss
        loss = loss.mean(dim=0)
        self.log("validation_loss", loss)

    def sample(self, n_shapes: int, n_points: int, steps: int = 100):
        condition = self.encoder.distribution.sample((n_shapes,)).to(self.device)
        assert condition.shape == (n_shapes, self.hparams.encoder_hparams["conditions"])

        noise = self.rectifier.distribution.sample((n_shapes, n_points)).to(self.device)
        assert noise.shape == (n_shapes, n_points, self.hparams.encoder_hparams["inputs"])

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)
        assert points.shape == noise.shape

        return points

    def sample_shapes(self, n_shapes: int, n_points: int, steps: int = 100):
        """ Sample with fixed latent noise """
        condition = self.encoder.distribution.sample((n_shapes,)).to(self.device)

        noise = self.rectifier.distribution.sample((1, n_points)).to(self.device)
        noise = utils.repeat_dim(noise, n_shapes, dim=0)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def sample_variations(self, n_shapes: int, n_points: int, steps: int = 100):
        """ Sample with fixed conditional """
        condition = self.encoder.distribution.sample((1,)).to(self.device)
        condition = utils.repeat_dim(condition, n_shapes, dim=0)

        noise = self.rectifier.distribution.sample((n_shapes, n_points)).to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def reconstruct(self, samples: torch.Tensor, steps: int = 100):
        n_shapes, n_points, n_dim = samples.shape
        samples = utils.normalize(samples, dim=1)
        samples = samples.to(self.device)

        condition = self.encoder.forward(samples)
        noise = self.rectifier.distribution.sample((n_shapes, n_points)).to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points
