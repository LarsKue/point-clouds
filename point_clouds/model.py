

import pytorch_lightning as lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import warnings

import utils
import losses

from encoder import Encoder
from integrators import EulerIntegrator, RK45Integrator
from neural_ode import NeuralODE


class PointCloudsModule(lightning.LightningModule):

    @property
    def default_hparams(self):
        return dict(
            input_dim=0,
            input_points=0,
            condition_dim=0,
            batch_size=1,
            sample_size=1,
            mmd_samples=None,
            optimizer="adam",
            learning_rate=1e-3,
            weight_decay=1e-5,
            encoder_widths=[[], []],
            rectifier_widths=[],
            encoder="deterministic",
            activation="relu",
            pooling="mean",
            integrator="euler",
            beta=0.5,
        )

    def __init__(self, train_data=None, val_data=None, test_data=None, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.encoder = self.configure_encoder()
        self.rectifier = self.configure_rectifier()

        self.encoder_distribution = self.configure_encoder_distribution()
        self.rectifier_distribution = self.configure_rectifier_distribution()

    def inference_step(self, batch, batch_idx=None):
        # (shapes, points, dim)
        points = utils.normalize(batch, dim=1)

        n_shapes, n_points, n_dim = points.shape
        n_samples = self.hparams.sample_size

        # (shapes, points, dim)
        target_noise = self.rectifier_distribution.sample((n_shapes, n_points)).to(self.device)

        # (shapes, condition_dim)
        condition = self.encoder.forward(points)

        # (samples * shapes, 1, 1)
        time = torch.rand(n_samples * n_shapes).to(self.device)
        time = utils.unsqueeze_as(time, points)

        target_noise = utils.repeat_dim(target_noise, n_samples, dim=0)
        points = utils.repeat_dim(points, n_samples, dim=0)
        condition = utils.repeat_dim(condition, n_samples, dim=0)

        # (samples * shapes, points, dim)
        interpolation = time * target_noise + (1 - time) * points

        # (samples * shapes, points, dim)
        v = self.rectifier.velocity(interpolation, time=time, condition=condition)

        # (samples *A shapes, points, dim)
        v_target = target_noise - points

        # (samples * shapes, condition_dim)
        mmd_samples = self.hparams.mmd_samples or n_shapes * n_samples
        encoder_noise = self.encoder_distribution.sample((mmd_samples,)).to(self.device)
        mmd_loss = losses.mmd(condition, encoder_noise)

        match self.hparams.encoder.lower():
            case "probabilistic":
                log_prob = torch.mean(self.encoder_distribution.log_prob(condition))
                Lambda = self.hparams.Lambda
                encoder_loss = Lambda * mmd_loss - (1 - Lambda) * log_prob
            case "deterministic":
                encoder_loss = mmd_loss
            case encoder:
                raise NotImplementedError(f"Unsupported Encoder: {encoder}")

        # (,)
        rectifier_loss = F.mse_loss(v, v_target)

        return encoder_loss, rectifier_loss

    def sample(self, n_shapes: int, n_points: int, steps: int = 100):
        condition = self.encoder_distribution.sample((n_shapes,)).to(self.device)

        noise = self.rectifier_distribution.sample((n_shapes, n_points)).to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def sample_shapes(self, n_shapes: int, n_points: int, steps: int = 100):
        """ Sample with fixed latent noise """
        condition = self.encoder_distribution.sample((n_shapes,)).to(self.device)

        noise = self.rectifier_distribution.sample((1, n_points)).to(self.device)
        noise = utils.repeat_dim(noise, n_shapes, dim=0)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def sample_variations(self, n_shapes: int, n_points: int, steps: int = 100):
        """ Sample with fixed conditional """
        condition = self.encoder_distribution.sample((1,)).to(self.device)
        condition = utils.repeat_dim(condition, n_shapes, dim=0)

        noise = self.rectifier_distribution.sample((n_shapes, n_points)).to(self.device)

        points, _time = self.rectifier.inverse(noise, condition=condition, steps=steps)

        return points

    def training_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)
        self.log("encoder_loss", encoder_loss)
        self.log("rectifier_loss", rectifier_loss)

        beta = self.hparams.beta
        loss = beta * encoder_loss + (1 - beta) * rectifier_loss
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx=None):
        encoder_loss, rectifier_loss = self.inference_step(batch, batch_idx)

        beta = self.hparams.beta
        loss = beta * encoder_loss + (1 - beta) * rectifier_loss
        self.log("validation_loss", loss)

    def configure_encoder(self):
        input_dim = self.hparams.input_dim
        input_points = self.hparams.input_points
        condition_dim = self.hparams.condition_dim
        pre_widths, post_widths = self.hparams.encoder_widths

        match self.hparams.activation.lower():
            case "relu":
                Activation = nn.ReLU
            case "elu":
                Activation = nn.ELU
            case "selu":
                Activation = nn.SELU
            case activation:
                raise NotImplementedError(f"Unsupported Activation: {activation}")

        pre_pool = nn.Sequential()

        input_layer = nn.Conv1d(in_channels=input_dim, out_channels=pre_widths[0], kernel_size=1)
        pre_pool.add_module("Pre-Pooling Input Layer", input_layer)
        pre_pool.add_module("Pre-Pooling Input Activation", Activation())

        for i in range(len(pre_widths) - 1):
            hidden_layer = nn.Conv1d(in_channels=pre_widths[i], out_channels=pre_widths[i + 1], kernel_size=1)
            pre_pool.add_module(f"Pre-Pooling Hidden Layer {i}", hidden_layer)
            pre_pool.add_module(f"Pre-Pooling Hidden Activation {i}", Activation())

        match self.hparams.encoder.lower():
            case "deterministic":
                out_channels = post_widths[0]
            case "probabilistic":
                out_channels = 2 * post_widths[0]
            case encoder:
                raise NotImplementedError(f"Unsupported Encoder: {encoder}")

        output_layer = nn.Conv1d(in_channels=pre_widths[-1], out_channels=out_channels, kernel_size=1)
        pre_pool.add_module("Pre-Pooling Output Layer", output_layer)

        match self.hparams.pooling.lower():
            case "mean":
                def pool(x):
                    return torch.mean(x, dim=-1)
            case "max":
                def pool(x):
                    return torch.max(x, dim=-1)[0]
            case pooling:
                raise NotImplementedError(f"Unsupported Pooling: {pooling}")

        post_pool = nn.Sequential()

        for i in range(len(post_widths) - 1):
            hidden_layer = nn.Linear(in_features=post_widths[i], out_features=post_widths[i + 1])
            post_pool.add_module(f"Post-Pooling Hidden Layer {i}", hidden_layer)
            post_pool.add_module(f"Post-Pooling Hidden Activation {i}", Activation())

        output_layer = nn.Linear(in_features=post_widths[-1], out_features=condition_dim)
        post_pool.add_module("Post-Pooling Output Layer", output_layer)

        return Encoder(pre_pool=pre_pool, pool=pool, post_pool=post_pool, kind=self.hparams.encoder)

    def test_encoder(self, n_shapes: int = 10, seed: int = 42):
        # test permutation invariance of the encoder
        n_points = self.hparams.input_points

        points = self.rectifier_distribution.sample((n_shapes, n_points)).to(self.device)

        torch.manual_seed(seed)
        condition = self.encoder.forward(points)

        perm = torch.randperm(points.shape[1])
        permuted = points[:, perm]

        torch.manual_seed(seed)
        permuted_condition = self.encoder.forward(permuted)

        close = torch.isclose(condition, permuted_condition)
        if not torch.all(close):
            max_dev = torch.max(torch.abs(condition - permuted_condition))
            n_close = close.sum()
            n = condition.numel()
            percentage_close = 100.0 * n_close / n
            warnings.warn(f"Encoder may be misspecified. {n_close}/{n} elements close ({percentage_close:.2f}%). "
                          f"Max deviation: {max_dev:.2e}")

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

        # input is x, time, condition
        input_layer = nn.Linear(in_features=input_dim + 1 + condition_dim, out_features=widths[0])
        network.add_module("Input Layer", input_layer)
        network.add_module("Input Activation", Activation())

        for i in range(len(widths) - 1):
            hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
            network.add_module(f"Hidden Layer {i}", hidden_layer)
            network.add_module(f"Hidden Activation {i}", Activation())

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
        loc = torch.zeros(self.hparams.condition_dim).cuda()
        scale = torch.ones(self.hparams.condition_dim).cuda()
        return torch.distributions.Normal(loc, scale)

    def configure_rectifier_distribution(self) -> torch.distributions.Distribution:
        loc = torch.zeros(self.hparams.input_dim).cuda()
        scale = torch.ones(self.hparams.input_dim).cuda()
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

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_loss", save_last=True),
            lightning.callbacks.LearningRateMonitor(),
        ]

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
