
import utils

from .encoder import Encoder
from .rectifier import Rectifier
from .trainable import Trainable


class PointCloudsModule(Trainable):

    @property
    def default_hparams(self):
        return super().default_hparams | dict(
            time_samples=1,
            mmd_samples=None,
            mmd_scales="all",
            beta=0.5,
            encoder_hparams=dict(),
            rectifier_hparams=dict(),
        )

    def __init__(self, *datasets, **hparams):
        super().__init__(*datasets, **hparams)

        self.encoder = Encoder(**self.hparams.encoder_hparams)
        self.hparams.encoder_hparams = self.encoder.hparams
        self.rectifier = Rectifier(**self.hparams.rectifier_hparams)
        self.hparams.rectifier_hparams = self.rectifier.hparams

        self.save_hyperparameters(self.hparams)

    def inference_step(self, batch, batch_idx=None):
        points = utils.normalize(batch, dim=1)

        n_shapes, n_points, dim = points.shape
        n_times = self.hparams.time_samples

        condition = self.encoder.forward(points)

        mmd_samples = self.hparams.mmd_samples or n_shapes * n_times

        encoder_loss = self.encoder.loss(condition, samples=mmd_samples, scales=self.hparams.mmd_scales)
        rectifier_loss = self.rectifier.loss(points, condition, samples=n_times)

        return encoder_loss, rectifier_loss

    def training_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)
        self.log("encoder_loss", encoder_loss.mean(dim=0))
        self.log("rectifier_loss", rectifier_loss.mean(dim=0))

        beta = self.hparams.beta
        loss = beta * encoder_loss + (1 - beta) * rectifier_loss
        loss = loss.mean(dim=0)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)

        beta = self.hparams.beta
        loss = beta * encoder_loss + (1 - beta) * rectifier_loss
        loss = loss.mean(dim=0)
        self.log("validation_loss", loss)

    def sample(self, n_shapes: int, n_points: int, steps: int = 100):
        condition = self.encoder.distribution.sample((n_shapes,)).to(self.device)

        noise = self.rectifier.distribution.sample((n_shapes, n_points)).to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

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
