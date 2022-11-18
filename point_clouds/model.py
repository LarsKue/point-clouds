

import pytorch_lightning as lightning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

import utils

from encoder import Encoder
from integrators import EulerIntegrator, RK45Integrator
from neural_ode import NeuralODE


class PointCloudsModule(lightning.LightningModule):

    @property
    def default_hparams(self):
        return dict(
            input_dim=0,
            condition_dim=0,
            batch_size=1,
            sample_size=1,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            encoder_widths=[],
            rectifier_widths=[],
            activation="relu"
        )

    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.encoder = self.configure_encoder()
        self.rectifier = self.configure_rectifier()

        self.encoder_distribution = self.configure_encoder_distribution()
        self.rectifier_distribution = self.configure_rectifier_distribution()

    def inference_step(self, batch, batch_idx=None):
        points = batch

        n_shapes, n_points, _ = points.shape

        target_noise = self.rectifier_distribution.sample((n_shapes, n_points))
        target_noise = target_noise.to(self.device)

        condition = self.encoder.forward(points)

        sample_size = self.hparams.sample_size
        time = torch.randn(n_shapes * sample_size)
        time = utils.unsqueeze_as(time, points)

        target_noise = utils.repeat_dim(target_noise, sample_size, dim=0)
        points = utils.repeat_dim(points, sample_size, dim=0)
        condition = utils.repeat_dim(condition, sample_size, dim=0)

        interpolation = time * target_noise + (1 - time) * points

        v = self.rectifier.velocity(interpolation, time=time, condition=condition)

        v_target = target_noise - points

        encoder_loss = -self.encoder_distribution.log_prob(condition)
        rectifier_loss = utils.mean_except_batch(torch.square(v - v_target))

        beta = self.hparams.beta
        loss = beta * encoder_loss + (1 - beta) * rectifier_loss

        return loss.mean(dim=0)

    def sample(self, n_shapes: int, n_points: int, steps: int = 100):
        condition = self.encoder_distribution.sample((n_shapes,))
        condition = condition.to(self.device)

        noise = self.rectifier_distribution.sample((n_shapes, n_points))
        noise = noise.to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def training_step(self, batch, batch_idx=None):
        loss = self.inference_step(batch, batch_idx)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx=None):
        loss = self.inference_step(batch, batch_idx)
        self.log("validation_loss", loss)

        return loss

    def configure_encoder(self):
        input_dim = self.hparams.input_dim
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

        network = nn.Sequential()

        input_layer = nn.Conv1d(in_channels=input_dim, out_channels=widths[0], kernel_size=1, padding="same")
        network.add_module("Input Layer", input_layer)

        activation = Activation()
        network.add_module("Input Activation", activation)

        for i in range(len(widths) - 1):
            hidden_layer = nn.Conv1d(in_channels=widths[i], out_channels=widths[i + 1], kernel_size=1, padding="same")
            network.add_module(f"Hidden Layer {i}", hidden_layer)

            activation = Activation()
            network.add_module(f"Hidden Activation {i}", activation)

        output_layer = nn.Conv1d(in_channels=widths[-1], out_channels=2 * condition_dim, kernel_size=1, padding="same")
        network.add_module("Output Layer", output_layer)

        return Encoder(network)

    def configure_rectifier(self):
        input_dim = self.hparams.input_dim
        condition_dim = self.hparams.condition_dim
        widths = self.hparams.rectifier_widths

        match self.hparams.activation.lower():
            case "relu":
                Activation = nn.ReLU
            case "elu":
                Activation = nn.ELU
            case "selu":
                Activation = nn.SELU
            case activation:
                raise NotImplementedError(f"Unsupported Activation: {activation}")

        network = nn.Sequential()

        # input is x, condition, time
        input_layer = nn.Linear(in_features=input_dim + condition_dim + 1, out_features=widths[0])
        network.add_module("Input Layer", input_layer)

        activation = Activation()
        network.add_module("Input Activation", activation)

        for i in range(len(widths) - 1):
            hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
            network.add_module(f"Hidden Layer {i}", hidden_layer)

            activation = Activation()
            network.add_module(f"Hidden Activation {i}", activation)

        # output is velocity
        output_layer = nn.Linear(in_features=widths[-1], out_features=input_dim)
        network.add_module("Output Layer", output_layer)

        integrator = self.configure_integrator()

        return NeuralODE(network, integrator)

    def configure_integrator(self):
        """ Configure and return integrator used for inference """
        match self.hparams.integrator.lower():
            case "euler":
                integrator = EulerIntegrator()
            case "rk45":
                integrator = RK45Integrator()
            case integrator:
                raise NotImplementedError(f"Unsupported Integrator: {integrator}")

        return integrator

    def configure_encoder_distribution(self) -> torch.distributions.Distribution:
        loc = torch.zeros(self.hparams.condition_dim)
        scale = torch.ones(self.hparams.condition_dim)
        return torch.distributions.Normal(loc, scale)

    def configure_rectifier_distribution(self) -> torch.distributions.Distribution:
        loc = torch.zeros(self.hparams.input_dim)
        scale = torch.ones(self.hparams.input_dim)
        return torch.distributions.Normal(loc, scale)

    def configure_optimizers(self):
        """
        Configure optimizers and LR schedulers for Lightning
        """
        match self.hparams.optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            case optimizer:
                raise NotImplementedError(f"Unsupported Optimizer: {optimizer}")

        return optimizer

    def train_dataloader(self):
        """
        Configure and return the train dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        """
        Configure and return the validation dataloader
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        """
        Configure and return the test dataloader
        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )
