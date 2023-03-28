import torch


def gaussian_kernel(x1: torch.Tensor, x2: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Kernel, also called Radial-Basis Function.
    Parameters
    ----------
    x1: Tensor of shape (N, D)
    x2: Tensor of shape (M, D)
    scales: Tensor of shape (K,)

    Returns
    -------
    Gaussian distance between x1 and x2
    """
    # x1: (N, D)
    # x2: (M, D)
    # scales: (K,)

    # (N, M, D)
    residuals = x1[:, None] - x2[None, :]

    # (N, M)
    norms = torch.sum(torch.square(residuals), dim=-1)

    # (N, M, K)
    exponent = norms[:, :, None] / (2.0 * scales[None, None, :])

    # (N, M)
    return torch.sum(torch.exp(-exponent), dim=-1)


def mmd(data: torch.Tensor, code: torch.Tensor, kernel=gaussian_kernel, scales: torch.Tensor = "all"):
    """
    Compute the Maximum-Mean-Discrepancy (MMD) between samples from two distributions
    Parameters
    ----------
    data: Samples from the data distrbution. Tensor of shape (N, D)
    code: Samples from the latent distribution. Tensor of shape (M, D)
    kernel: Kernel distance function to use
    scales: Characteristic scales used in the kernel function. Tensor of shape (K,)

    Returns
    -------
    mmd: Maximum-Mean-Discrepancy between data and code
    """

    if scales == "all":
        scales = torch.logspace(-6, 6, 30)

    code = code.to(data.device)
    scales = scales.to(data.device)

    l1 = torch.mean(kernel(data, data, scales=scales))
    l2 = torch.mean(kernel(code, code, scales=scales))
    l3 = torch.mean(kernel(data, code, scales=scales))

    # out: (,)
    return l1 + l2 - 2.0 * l3


def _gaussian_kernel_matrix(x, y, sigmas):
    sigmas = torch.Tensor(sigmas)[None, :].to(x.device)
    dist = ((x[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1)[:, :, None]
    beta = 1. / (2. * sigmas)
    s = torch.matmul(dist, beta)
    k = torch.exp(-s).sum(dim=-1)
    return k


def _mmd(embedding, n_samples: int = None):
    if n_samples is None:
        z = torch.randn_like(embedding)
    else:
        z = torch.randn(n_samples, embedding.shape[1]).to(embedding.device)

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]

    loss = torch.mean(_gaussian_kernel_matrix(embedding, embedding, sigmas))
    loss += torch.mean(_gaussian_kernel_matrix(z, z, sigmas))
    loss -= 2 * torch.mean(_gaussian_kernel_matrix(embedding, z, sigmas))
    return loss
