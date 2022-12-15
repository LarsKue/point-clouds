import torch
import torch.nn as nn

import utils
import numpy as np


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
        # TODO: use torch.topk
        sort = torch.argsort(x, dim=int(self.dim), descending=True)
        indices = torch.arange(self.outputs).to(x.device)
        multimax = torch.index_select(sort, int(self.dim), indices)

        return torch.gather(x, int(self.dim), multimax)


class MAB(nn.Module):
    """ Multihead Attention Block as described in arXiV:1810.00825 """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        d = in_features
        dq = hidden_features
        dv = out_features

        self.linear_q = nn.Linear(d, dq)
        self.linear_k = nn.Linear(d, dq)
        self.linear_v = nn.Linear(d, dv)

        # slight difference: the paper makes some dimensionality mistakes
        self.linear_x = nn.Linear(d, dv)

        self.att = nn.MultiheadAttention(dq, dq, dv, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dv, dv),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform the forward computation
        Parameters
        ----------
        x: Query Tensor of shape (N, D)
        y: Keys Tensor of shape (M, D)

        Returns
        -------
        MAB Tensor of shape (N, Dv)
        """
        # m == nv
        # q: (n, dq)
        # k: (nv, dq)
        # v: (nv, dv)
        q = self.linear_q(x)
        k = self.linear_v(y)
        v = self.linear_k(y)

        # x: (n, dv)
        x = self.linear_x(x)

        # att: (n, dv)
        att, _weights = self.att(q, k, v, need_weights=False)

        # h: (n, dv)
        h = self.ln1(x + att)

        # output: (n, dv)
        return self.ln2(h + self.ff(h))


class MAB(nn.Module):
    """ Multihead Attention Block as described in arXiV:1810.00825 """
    def __init__(self, embed_dim: int, kdim: int, vdim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, kdim=kdim, vdim=vdim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, kdim)
        self.linear_v = nn.Linear(embed_dim, vdim)

        self.ln1 = nn.LayerNorm(vdim)
        self.ln2 = nn.LayerNorm(vdim)

        self.ff = nn.Sequential(
            nn.Linear(vdim, vdim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d)
        # y: (m, d)

        # q: (n, dq)
        # k: (nv, dq)
        # v: (nv, dv)
        q = self.linear_q(x)
        k = self.linear_k(y)
        v = self.linear_v(y)


        # att: (n, dv)
        att, _weights = self.att(q, k, v, need_weights=False)

        # error: (n, d) + (n, dv)
        h = self.ln1(x + att)
        return self.ln2(h + self.ff(h))


class SAB(nn.Module):
    """ Set Attention Block as described in arXiV:1810.00825 """
    def __init__(self, in_features: int, out_features: int, heads: int):
        super().__init__()
        self.mab = MAB(in_features, in_features, out_features, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x)


class ISAB(nn.Module):
    """ Induced Set Attention Block as described in arXiV:1810.00825 """
    def __init__(self, in_features: int, out_features: int, num_heads: int, inducing_points: int):
        super().__init__()
        self.mab1 = MAB(out_features, in_features, out_features, num_heads)
        self.mab2 = MAB(in_features, out_features, out_features, num_heads)

        self.inducing_points = nn.Parameter(torch.Tensor(1, inducing_points, out_features))
        nn.init.xavier_uniform_(self.inducing_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = utils.repeat_dim(self.inducing_points, x.shape[0], dim=0)
        h = self.mab1(i, x)
        return self.mab2(x, h)


class PMA(nn.Module):
    """ Pooling by Multihead Attention as described in arXiV:1810.00825 """
    def __init__(self, dim: int, num_heads: int, num_seeds: int):
        super().__init__()

        # multihead attention block
        self.mab = MAB(dim, dim, dim, num_heads)

        # seed-vector S
        self.seed = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = utils.repeat_dim(self.seed, x.shape[0], dim=0)
        return self.mab(s, x)

