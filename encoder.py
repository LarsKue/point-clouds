
import torch
import torch.nn as nn

import utils


class Encoder(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor):
        # x.shape == (shapes, points, dim)
        x = x.movedim(1, -1)
        z = self.network.forward(x)

        mu, sigma = torch.tensor_split(z, 2, dim=1)

        out = mu + sigma * torch.randn_like(sigma)

        # TODO: learnable pooling (fc)
        # global average pooling
        pool = torch.mean(out, dim=-1)

        # out.shape == (shapes, condition_dim)
        return pool
