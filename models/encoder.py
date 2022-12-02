
import pytorch_lightning as lightning
import torch
import torch.nn as nn

import warnings

import losses
from .utils import make_conv, make_dense
from .pools import *


class Encoder(lightning.LightningModule):
    @property
    def default_hparams(self):
        return dict(
            inputs=0,
            points=0,
            conditions=0,
            kind="deterministic",
            dropout=None,
            batchnorm=False,
            widths=[[], []],
            activation="relu",
            pooling="mean",
            Lambda=0.5,
        )

    def __init__(self, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams

        self.save_hyperparameters(hparams)

        self.encode = self.configure_encode()
        self.post_encode = self.configure_post_pool()

        # self.pre_pool = self.configure_pre_pool()
        # self.pool = self.configure_pool()
        # self.post_pool = self.configure_post_pool()

        self.distribution = self.configure_distribution()

        # self._test_permutation_equivariance()
        self._test_permutation_invariance()

    def forward(self, x: torch.Tensor):
        """
        Forward Inference Step
        Parameters
        ----------
        x: Tensor of shape (shapes, points, dim)

        Returns
        -------
        condition: Tensor of shape (shapes, conditions)
        """
        x = x.movedim(1, -1)

        print(f"{x.shape=}")
        z = self.encode(x)
        z = self.post_encode(z)

        return z

        # z.shape == (shapes, ..., points), permutation-equivariant
        z = self.pre_pool(x)

        # pooled.shape == (shapes, ...), permutation-invariant
        pooled = self.pool(z)

        match self.hparams.kind.lower():
            case "deterministic":
                pass
            case "probabilistic":
                # mu.shape == sigma.shape == (shapes, conditions)
                mu, sigma = torch.tensor_split(pooled, 2, dim=1)

                # pooled.shape == (shapes, conditions)
                pooled = mu + sigma * torch.randn_like(sigma)
            case kind:
                raise NotImplementedError(f"Unsupported Encoder: {kind}")

        # out.shape == (shapes, conditions)
        out = self.post_pool(pooled)

        # out.shape == (shapes, conditions)
        return out

    def loss(self, condition: torch.Tensor, samples: int, scales: torch.Tensor = "all"):
        """
        Compute the loss for the encoded condition compared with a noise sample
        Parameters
        ----------
        condition: The condition output of the encoder. Tensor of shape (shapes, conditions)
        samples: How many samples to use for the MMD
        scales: Optional MMD scales to use

        Returns
        -------
        loss: Tensor of shape ()
        """

        noise = self.distribution.sample((samples,)).to(self.device)
        mmd = losses.mmd(condition, noise, scales=scales)

        match self.hparams.kind.lower():
            case "deterministic":
                return mmd
            case "probabilistic":
                log_prob = torch.mean(self.distribution.log_prob(condition))
                Lambda = self.hparams.Lambda
                return Lambda * mmd - (1 - Lambda) * log_prob
            case kind:
                raise NotImplementedError(f"Unsupported kind: {kind}")

    def configure_pre_pool(self):
        """ Configure and return the permutation-equivariant pre-pool network """
        inputs = self.hparams.inputs
        widths = self.hparams.widths[0]

        match self.hparams.kind.lower():
            case "deterministic":
                outputs = widths[-1]
            case "probabilistic":
                # x ~ N(mu, sigma)
                outputs = 2 * widths[-1]
            case kind:
                raise NotImplementedError(f"Unsupported kind: {kind}")

        network = make_conv(
            widths=[inputs, *widths, outputs],
            activation=self.hparams.activation,
            batchnorm=self.hparams.batchnorm,
            dropout=self.hparams.dropout,
        )

        return network

    def configure_encode(self):
        inputs = self.hparams.inputs
        # widths = self.hparams.widths[0]

        # TODO: runs out of memory, yields error "no valid convolution algorithm in cudnn"
        network = nn.Sequential(
            nn.Conv1d(in_channels=inputs, out_channels=8, kernel_size=1),
            nn.ReLU(),
            GlobalMultimaxPool1d(outputs=32),
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=1),
            nn.Flatten()
        )

        # network = nn.Sequential(
        #     # (3, 2048)
        #     nn.Conv1d(in_channels=inputs, out_channels=4, kernel_size=1),
        #     nn.ReLU(),
        #     # (16, 2048)
        #     GlobalMultimaxPool1d(outputs=512),
        #     # (16, 512)
        #     nn.Conv1d(in_channels=4, out_channels=8, kernel_size=1),
        #     nn.ReLU(),
        #     # (32, 512)
        #     GlobalMultimaxPool1d(outputs=128),
        #     # (32, 128)
        #     nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1),
        #     nn.ReLU(),
        #     # (64, 128)
        #     GlobalMultimaxPool1d(outputs=64),
        #     # (64, 64)
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1),
        #     nn.ReLU(),
        #     # (128, 64)
        #     GlobalMultimaxPool1d(outputs=32),
        #     # (128, 32)
        #     nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1),
        #     nn.ReLU(),
        #     # (128, 32)
        #     nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1),
        #     nn.ReLU(),
        #     # (64, 32)
        #     nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1),
        #     # (32, 32)
        #     nn.Flatten(),
        #     # (1024,)
        # )

        return network

    def configure_pool(self):
        """ Configure and return the pooling method """
        pool = nn.Sequential()
        match self.hparams.pooling.lower():
            case "max":
                pool.add_module("Max Pool", GlobalMaxPool1d())
            case "mean":
                pool.add_module("Average Pool", GlobalAvgPool1d())
            case "multimax":
                pool.add_module("Multimax Pool", GlobalMultimaxPool1d(outputs=self.hparams.pools))
            case "stats":
                pool.add_module("Statistics Pool", GlobalStatisticsPool1d())
            case pooling:
                raise NotImplementedError(f"Unsupported Pooling: {pooling}")

        pool.add_module("Flatten", nn.Flatten())

        return pool

    def configure_post_pool(self):
        """ Configure and return the post pool network """
        inputs = self.hparams.pools * self.hparams.widths[0][-1]
        conditions = self.hparams.conditions
        widths = self.hparams.widths[1]

        network = make_dense(
            widths=[inputs, *widths, conditions],
            activation=self.hparams.activation,
            batchnorm=self.hparams.batchnorm,
            dropout=self.hparams.dropout,
        )

        return network

    def configure_distribution(self):
        loc = torch.zeros(self.hparams.conditions).cuda()
        scale = torch.ones(self.hparams.conditions).cuda()
        return torch.distributions.Normal(loc, scale)

    @torch.no_grad()
    def _test_permutation_equivariance(self, shapes=10, points=128):
        train = self.training
        self.eval()
        permutation = torch.randperm(points)
        inverse_permutation = torch.argsort(permutation)

        x = torch.randn(shapes, self.hparams.inputs, points)
        xp = x[:, :, permutation]

        z = self.pre_pool.forward(x)
        zp = self.pre_pool.forward(xp)

        zz = zp[:, :, inverse_permutation]

        close = torch.isclose(z, zz, atol=1e-7)
        if not torch.all(close):
            max_dev = torch.max(torch.abs(z - zz))
            n_close = close.sum()
            n = z.numel()
            percentage_close = 100.0 * n_close / n
            warnings.warn(f"Encoder Pre-Pool may not be permutation-equivariant. "
                          f"{n_close}/{n} elements close ({percentage_close:.2f}%). "
                          f"Max deviation: {max_dev:.2e}")

        self.train(train)

    @torch.no_grad()
    def _test_permutation_invariance(self, shapes=10):
        train = self.training
        self.eval()
        permutation = torch.randperm(self.hparams.points)

        x = torch.randn(shapes, self.hparams.points, self.hparams.inputs)
        xp = x[:, permutation, :]

        seed = torch.seed()
        torch.manual_seed(0)
        z = self.forward(x)
        torch.manual_seed(0)
        zp = self.forward(xp)
        torch.manual_seed(seed)

        close = torch.isclose(z, zp, atol=1e-7)
        if not torch.all(close):
            max_dev = torch.max(torch.abs(z - zp))
            n_close = close.sum()
            n = z.numel()
            percentage_close = 100.0 * n_close / n
            warnings.warn(f"Encoder may not be permutation-invariant. "
                          f"{n_close}/{n} elements close ({percentage_close:.2f}%). "
                          f"Max deviation: {max_dev:.2e}")

        self.train(train)
