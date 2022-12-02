import torch
import torch.nn as nn
import torchsort


class GlobalAvgPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):
        return torch.mean(x, dim=int(self.dim))


class GlobalMaxPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):
        return torch.max(x, dim=int(self.dim))[0]


class GlobalStatisticsPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):
        mean = torch.mean(x, dim=int(self.dim))
        var = torch.var(x, dim=int(self.dim))

        return torch.stack([mean, var], dim=int(self.dim))


class GlobalMultimaxPool1d(nn.Module):
    def __init__(self, outputs: int = 1, dim: int = -1):
        super().__init__()
        self.register_buffer("outputs", torch.tensor(outputs, dtype=torch.int32))
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):

        z = torch.stack([torchsort.soft_sort(x[i], regularization_strength=0.1) for i in range(len(x))])
        # z = torchsort.soft_sort(x, regularization_strength=0.1)

        indices = z.shape[int(self.dim)] - torch.arange(self.outputs) - 1
        indices = indices.to(z.device)
        return torch.index_select(z, int(self.dim), indices)
