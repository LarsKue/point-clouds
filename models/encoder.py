
import pytorch_lightning as lightning
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

import warnings

import losses
from .utils import get_activation
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
            widths=[[], []],
            activation="relu",
            checkpoints=None,
            Lambda=0.5,
        )

    def __init__(self, **hparams):
        super().__init__()
        hparams = self.default_hparams | hparams

        self.save_hyperparameters(hparams)

        self.network = self.configure_network()

        self.distribution = self.configure_distribution()

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
        if not self.training or not self.hparams.checkpoints:
            output = self.network(x)
        else:
            if self.hparams.checkpoints is True:
                output = checkpoint(
                    self.network,
                    x,
                    use_reentrant=False,
                )
            else:
                output = checkpoint_sequential(
                    functions=self.network,
                    segments=self.hparams.checkpoints,
                    input=x
                )

        match self.hparams.kind.lower():
            case "deterministic":
                pass
            case "probabilistic":
                mu, sigma = torch.tensor_split(output, 2, dim=-1)
                output = mu + sigma * torch.randn_like(sigma)

        return output

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

    def configure_network(self):
        inputs = self.hparams.inputs
        conditions = self.hparams.conditions
        dropout = self.hparams.dropout

        Activation = get_activation(self.hparams.activation)

        network = nn.Sequential(
            nn.Linear(inputs, 128),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(128, 256),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(256, 512),
            Activation(),

            GlobalMultimaxPool1d(1024, dim=1),

            nn.Linear(512, 512),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(512, 512),
            Activation(),

            GlobalMultimaxPool1d(512, dim=1),

            nn.Linear(512, 512),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(512, 512),
            Activation(),

            GlobalMultimaxPool1d(256, dim=1),

            nn.Linear(512, 256),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(256, 256),
            Activation(),

            nn.Dropout(dropout),
            nn.Linear(256, 256),
            Activation(),

            GlobalAvgPool1d(dim=1),

            nn.Linear(256, 256),
            Activation(),

            nn.Linear(256, conditions),
        )

        # network = nn.Sequential(
        #     # (batch, 2048, 3)
        #     nn.Linear(inputs, 128),
        #     nn.ReLU(),
        #     # (batch, 2048, 128)
        #     ISAB(128, 128, num_heads=4, inducing_points=16),
        #     ISAB(128, 128, num_heads=4, inducing_points=16),
        #     nn.Dropout(dropout),
        #     PMA(128, num_heads=4, num_seeds=16),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     ISAB(256, 256, num_heads=4, inducing_points=32),
        #     ISAB(256, 256, num_heads=4, inducing_points=32),
        #     nn.Dropout(dropout),
        #     PMA(256, num_heads=4, num_seeds=16),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, conditions),
        #     GlobalAvgPool1d(dim=1),
        # )

        x = torch.randn(10, self.hparams.points, self.hparams.inputs)
        z = network.forward(x)
        assert z.shape == (10, conditions)

        return network

    def configure_distribution(self):
        loc = torch.zeros(self.hparams.conditions).cuda()
        scale = torch.ones(self.hparams.conditions).cuda()
        return torch.distributions.Normal(loc, scale)

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
