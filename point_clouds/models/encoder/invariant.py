import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_trainable.hparams import HParams, Choice
from lightning_trainable.modules import DenseModule, DenseModuleHParams

from .pools import GlobalAvgPool1d

from point_clouds.utils import temporary_seed


class InvariantEncoderHParams(DenseModuleHParams):
    equivariant_module: nn.Module
    pool: str = Choice("mean", "sum")


class InvariantEncoder(DenseModule):
    hparams: InvariantEncoderHParams

    def __init__(self, hparams: InvariantEncoderHParams | dict):
        super().__init__(hparams)

        self.encoder = self.hparams.equivariant_module
        self.pooler = self.configure_pooler()

    def forward(self, batch) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch[0]
            assert torch.is_tensor(batch)

        equivariant_encoding = self.encoder(batch)
        pooled_encoding = self.pooler(equivariant_encoding)
        transformed_encoding = self.network(pooled_encoding)

        return transformed_encoding

    def configure_pooler(self) -> nn.Module:
        return GlobalAvgPool1d(dim=1)

    @temporary_seed(0)
    def test_equivariance(self, shape: torch.Size | tuple):
        batch_size, samples, inputs = shape

        permutation = torch.randperm(samples)
        inverse_permutation = torch.argsort(permutation)

        x = torch.randn(batch_size, samples, inputs)
        xp = x[:, permutation, :]

        with temporary_seed(0):
            z = self.encoder(x)

        with temporary_seed(0):
            zp = self.encoder(xp)

        zz = zp[:, inverse_permutation, :]

        assert torch.allclose(z, zz, atol=1e-7), f"Equivariance not met. MSE = {F.mse_loss(z, zz):.2e}"

    @temporary_seed(0)
    def test_invariance(self, shape: torch.Size | tuple):
        batch_size, samples, inputs = shape

        permutation = torch.randperm(samples)

        x = torch.randn(batch_size, samples, inputs)
        xp = x[:, permutation, :]

        with temporary_seed(0):
            z = self(x)

        with temporary_seed(0):
            zp = self(xp)

        assert torch.allclose(z, zp, atol=1e-7), f"Invariance not met. MSE = {F.mse_loss(z, zp):.2e}"
