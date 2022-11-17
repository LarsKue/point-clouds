

import pytorch_lightning as lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from rectflow import RectifyingFlow


class PointCloudsModule(lightning.LightningModule):

    # TODO: maybe sample size (but is kinda complicated)

    @property
    def default_hparams(self):
        return dict(
            input_shape=(),
            batch_size=1,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            integrator="euler",
            encoder_widths=[],
            conditioner_widths=[],
            sampler_widths=[],
            activation="relu",
        )

    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.encoder = self.configure_encoder()
        self.sampler = self.configure_sampler()

        self.sample_distribution = self.configure_sample_distribution()
        self.condition_distribution = self.configure_condition_distribution()

    def configure_encoder(self):
        points, dim = self.hparams.input_shape
        condition_dim = self.hparams.condition_dim
        widths = self.hparams.encoder_widths

        match self.hparams.activation.lower():
            case "relu":
                Activation = nn.ReLU
            case "elu":
                Activation = nn.ELU
            case "selu":
                Activation = nn.SELU
            case activation:
                raise NotImplementedError(f"Unsupported Activation: {activation}")

        encoder = nn.Sequential()

        input_layer = nn.Conv1d(in_channels=points, out_channels=widths[0], kernel_size=1, padding="same")
        encoder.add_module("Input Layer", input_layer)

        activation = Activation()
        encoder.add_module("Input Activation", activation)

        for i in range(len(widths) - 1):
            hidden_layer = nn.Conv1d(in_channels=widths[i], out_channels=widths[i + 1], kernel_size=1, padding="same")
            encoder.add_module(f"Hidden Layer {i}", hidden_layer)

            activation = Activation()
            encoder.add_module(f"Hidden Activation {i}", activation)

        output_layer = nn.Conv1d(in_channels=widths[-1], out_channels=2 * condition_dim, kernel_size=1, padding="same")
        encoder.add_module("Output Layer", output_layer)

        return encoder

    def configure_sampler(self):
        hparams = dict(
            input_shape=self.hparams.input_shape,
            integrator=self.hparams.integrator,
            network_widths=self.hparams.sampler_widths,
            activation=self.hparams.activation,
        )

        return RectifyingFlow(**hparams)

    def training_step(self, batch, batch_idx=None):
        points, noise = batch

        z = self.encoder.forward(points)
        mu, sigma = torch.chunk(z, 2, dim=1)
        condition = mu + sigma * torch.randn_like(sigma)

        encoder_loss = -self.condition_distribution.log_prob(condition)

        # x0: noise
        # x1: condition
        # such that forward step of rectflow is noise -> condition

        time = torch.rand(condition.shape[0])
        xt = time * condition + (1 - time) * noise

        sampler_predicted = self.sampler.velocity(x=xt, time=time)
        sampler_target = condition - noise
        sampler_loss = torch.mean(torch.square(sampler_predicted - sampler_target), dim=1)

        loss = torch.mean(encoder_loss + sampler_loss, dim=0)

        return loss

    def sample(self, n: int):
        condition = self.condition_distribution.sample(n)

        sample_noise = self.sample_distribution.sample(n)
        # TODO: conditional rectflow
        sample = self.sampler.forward(sample_noise, condition=condition)

        return sample
