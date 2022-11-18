
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor):
        z = self.network.forward(x)

        mu, sigma = torch.tensor_split(z, 2, dim=1)

        return mu + sigma * torch.randn_like(sigma)
