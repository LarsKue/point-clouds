
import pytorch_lightning as lightning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from integrators import EulerIntegrator, RK45Integrator
import utils


class RectifyingFlow(lightning.LightningModule):

    @property
    def default_hparams(self):
        return dict(
            input_shape=(),
            batch_size=1,
            sample_size=1,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            integrator="euler",
            network_widths=[],
            kernel_sizes=[],
        )

    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        # merge hparams with merge operator (python 3.9+)
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.network = self.configure_network()
        self.integrator = self.configure_integrator()

    def forward(self, x: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """ Full Forward Inference """
        device = x.device

        x = x.to(self.device)
        dt = 1.0 / steps
        z = self.integrator.solve(f=self.network, x0=x, dt=dt, steps=steps)

        return z.to(device)

    def inverse(self, z: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """ Full Inverse Inference """
        device = z.device

        z = z.to(self.device)
        dt = -1.0 / steps
        x = self.integrator.solve(f=self.network, x0=z, dt=dt, steps=steps)

        return x.to(device)

    def inference_step(self, batch, batch_idx=None):
        """ Compute predicted and target force for a given batch """
        x0, x1 = batch

        sample_size = self.hparams.sample_size
        batch_size = x0.shape[0]
        input_shape = self.hparams.input_shape

        t = torch.rand(sample_size, batch_size).to(self.device)
        t = utils.unsqueeze_to(t, x0.dim() + 1, side="right")

        xt = t * x1 + (1 - t) * x0
        xt = xt.reshape(sample_size * batch_size, *input_shape)

        predicted = self.network.forward(xt)
        target = x1 - x0
        target = utils.repeat_as(target, predicted)

        return predicted, target

    def training_step(self, batch, batch_idx=None):
        predicted, target = self.inference_step(batch, batch_idx)

        mse = F.mse_loss(predicted, target)
        self.log("training_loss", mse)

        return mse

    def validation_step(self, batch, batch_idx):
        predicted, target = self.inference_step(batch, batch_idx)

        mse = F.mse_loss(predicted, target)
        self.log("validation_loss", mse)

    def test_step(self, batch, batch_idx):
        predicted, target = self.inference_step(batch, batch_idx)

        mse = F.mse_loss(predicted, target)
        self.log("test_loss", mse)

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_loss", save_last=True),
            lightning.callbacks.LearningRateMonitor(),
        ]

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

    def configure_network(self):
        """ Detect input shape and construct according network """
        match self.hparams.input_shape:
            case (int(dim),):
                network = self.configure_dense_network(dim, dim)
            case (int(channels), int(height), int(width)):
                network = self.configure_conv_network(channels, channels)
            case shape:
                raise NotImplementedError(f"Unsupported Input Shape: {shape}")

        return network

    def configure_dense_network(self, in_features, out_features):
        """ Construct a dense network from given hparams """
        assert len(self.hparams.network_widths) > 0
        network = nn.Sequential()

        in_layer = nn.Linear(in_features=in_features, out_features=self.hparams.network_widths[0])
        network.add_module("Input Layer", in_layer)

        relu = nn.ReLU()
        network.add_module(f"Input Activation", relu)

        for i in range(len(self.hparams.network_widths) - 1):
            linear = nn.Linear(in_features=self.hparams.network_widths[i], out_features=self.hparams.network_widths[i + 1])
            network.add_module(f"Hidden Layer {i}", linear)
            relu = nn.ReLU()
            network.add_module(f"Hidden Activation {i}", relu)

        out_layer = nn.Linear(in_features=self.hparams.network_widths[-1], out_features=out_features)
        network.add_module("Output Layer", out_layer)

        return network

    def configure_conv_network(self, in_channels, out_channels):
        """ Construct a convolutional network from given hparams """
        assert len(self.hparams.network_widths) > 0
        assert len(self.hparams.kernel_sizes) == len(self.hparams.network_widths) + 1
        network = nn.Sequential()

        in_layer = nn.Conv2d(in_channels=in_channels, out_channels=self.hparams.network_widths[0], kernel_size=self.hparams.kernel_sizes[0], padding="same")
        network.add_module("Input Layer", in_layer)

        relu = nn.ReLU()
        network.add_module(f"Input Activation", relu)

        for i in range(len(self.hparams.network_widths) - 1):
            conv = nn.Conv2d(in_channels=self.hparams.network_widths[i], out_channels=self.hparams.network_widths[i + 1], kernel_size=self.hparams.kernel_sizes[i + 1], padding="same")
            network.add_module(f"Hidden Layer {i}", conv)
            relu = nn.ReLU()
            network.add_module(f"Hidden Activation {i}", relu)

        out_layer = nn.Conv2d(in_channels=self.hparams.network_widths[-1], out_channels=out_channels, kernel_size=self.hparams.kernel_sizes[-1], padding="same")
        network.add_module("Output Layer", out_layer)

        return network

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
