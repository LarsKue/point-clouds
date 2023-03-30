import torch
import torch.nn as nn


class GlobalAvgPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):
        return torch.mean(x, dim=int(self.dim))


class GlobalSumPool1d(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer("dim", torch.tensor(dim, dtype=torch.int32))

    def forward(self, x):
        return torch.sum(x, dim=int(self.dim))


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
        # TODO: use torch.topk
        sort = torch.argsort(x, dim=int(self.dim), descending=True)
        indices = torch.arange(self.outputs).to(x.device)
        multimax = torch.index_select(sort, int(self.dim), indices)

        return torch.gather(x, int(self.dim), multimax)
