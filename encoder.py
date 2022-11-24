
import torch
import torch.nn as nn

import utils


class Encoder(nn.Module):
    def __init__(self, *, pre_pool=None, pool=None, post_pool=None, kind="probabilistic"):
        super().__init__()
        self.pre_pool = pre_pool or (lambda x: x)
        self.pool = pool or (lambda x: torch.mean(x, dim=-1))
        self.post_pool = post_pool or (lambda x: x)
        self.kind = kind

    def forward(self, x: torch.Tensor):
        # x.shape == (shapes, points, dim)
        x = x.movedim(1, -1)

        # z.shape == (shapes, ..., points)
        z = self.pre_pool(x)

        # pooled.shape == (shapes, ...)
        pooled = self.pool(z)

        match self.kind.lower():
            case "probabilistic":
                # mu.shape == sigma.shape == (shapes, condition_dim)
                mu, sigma = torch.tensor_split(pooled, 2, dim=1)

                # pooled.shape == (shapes, condition_dim)
                pooled = mu + sigma * torch.randn_like(sigma)
            case "deterministic":
                pass
            case kind:
                raise NotImplementedError(f"Unsupported Encoder: {kind}")

        # out.shape == (shapes, condition_dim)
        out = self.post_pool(pooled)

        # out.shape == (shapes, condition_dim)
        return out
