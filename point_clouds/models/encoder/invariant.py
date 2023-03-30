import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_trainable.hparams import HParams
from lightning_trainable.modules import HParamsModule, DenseModuleHParams

from point_clouds.utils import temporary_seed
from .pools import GlobalAvgPool1d, GlobalMultimaxPool1d, GlobalSumPool1d


class InvariantEncoderHParams(HParams):
    layers: list
    """
    Example: 
        layers = [
        {"linear": 128},
        {"linear": 256},
        {"multimax": 1024},
        "mean"
    ]
    """


class InvariantEncoder(HParamsModule):
    hparams: InvariantEncoderHParams

    def __init__(self, hparams: DenseModuleHParams | dict):
        super().__init__(hparams)

        self.network = self.configure_network()

    def forward(self, batch) -> torch.Tensor:
        if not torch.is_tensor(batch):
            batch = batch[0]
            assert torch.is_tensor(batch)

        return self.network(batch).unsqueeze(1)

    def configure_network(self):
        network = nn.Sequential()

        for layer in self.hparams.layers:
            match layer:
                case {"linear": width}:
                    network.append(nn.LazyLinear(width))
                case {"multimax": width}:
                    network.append(GlobalMultimaxPool1d(width, dim=1))
                case "mean":
                    network.append(GlobalAvgPool1d(dim=1))
                case "sum":
                    network.append(GlobalSumPool1d(dim=1))
                case "relu":
                    network.append(nn.ReLU())
                case "gelu":
                    network.append(nn.GELU())
                case other:
                    raise NotImplementedError(f"Unrecognized layer: {other}")

        return network

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
