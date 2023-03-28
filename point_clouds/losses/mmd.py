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


def mmd_loss(samples: torch.Tensor, target: torch.Tensor, kernel=gaussian_kernel, scales: torch.Tensor | str | int = "auto"):
    """
    Compute the Maximum-Mean-Discrepancy (MMD) between samples from two distributions
    Parameters
    ----------
    samples: Samples from the target distrbution. Tensor of shape (N, D)
    target: Samples from the actual distribution. Tensor of shape (M, D)
    kernel: Kernel distance function to use
    scales: Characteristic scales used in the kernel function. Tensor of shape (K,)

    Returns
    -------
    mmd: Maximum-Mean-Discrepancy between data and code
    """
    match scales:
        case int():
            scales = torch.logspace(-3, 3, scales)
        case str():
            assert scales == "auto"
            scales = torch.logspace(-3, 3, 30)
        case torch.Tensor():
            pass

    target = target.to(samples.device)
    scales = scales.to(samples.device)

    l1 = torch.mean(kernel(samples, samples, scales=scales))
    l2 = torch.mean(kernel(target, target, scales=scales))
    l3 = torch.mean(kernel(samples, target, scales=scales))

    # out: (,)
    return l1 + l2 - 2.0 * l3
