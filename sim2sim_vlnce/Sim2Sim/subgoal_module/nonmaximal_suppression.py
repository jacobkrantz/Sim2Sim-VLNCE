"""Adapted from https://github.com/batra-mlp-lab/vln-sim2real"""

import torch
from torch import Tensor


def neighborhoods(
    mu: Tensor,
    x_range: float,
    y_range: float,
    sigma: float,
    circular_x: bool = True,
    gaussian: bool = False,
):
    """Generate masks centered at mu of the given x and y range with the
        origin in the centre of the output
    Inputs:
        mu: tensor (N, 2)
    Outputs:
        tensor (N, y_range, s_range)
    """
    x_mu = mu[:, 0].unsqueeze(1).unsqueeze(1)
    y_mu = mu[:, 1].unsqueeze(1).unsqueeze(1)
    # Generate bivariate Gaussians centered at position mu
    x = (
        torch.arange(start=0, end=x_range, device=mu.device, dtype=mu.dtype)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    y = (
        torch.arange(start=0, end=y_range, device=mu.device, dtype=mu.dtype)
        .unsqueeze(1)
        .unsqueeze(0)
    )
    y_diff = y - y_mu
    x_diff = x - x_mu
    if circular_x:
        x_diff = torch.min(torch.abs(x_diff), torch.abs(x_diff + x_range))
    if gaussian:
        output = torch.exp(
            -0.5 * ((x_diff / sigma) ** 2 + (y_diff / sigma) ** 2)
        )
    else:
        output = 0.5 * (torch.abs(x_diff) <= sigma).type(mu.dtype) + 0.5 * (
            torch.abs(y_diff) <= sigma
        ).type(mu.dtype)
        output[output < 1] = 0
    return output


def nms(
    pred: Tensor,
    sigma: float,
    thresh: float,
    max_predictions: int,
    gaussian: bool = False,
):
    """Non-maximal suppression (NMS). Input (batch_size, 1, height, width)"""

    shape = pred.shape
    output = torch.zeros_like(pred)
    flat_pred = pred.reshape((shape[0], -1))
    supp_pred = pred.clone()
    flat_output = output.reshape((shape[0], -1))
    for _ in range(max_predictions):
        # Find and save max
        flat_supp_pred = supp_pred.reshape((shape[0], -1))
        _, ix = torch.max(flat_supp_pred, dim=1)
        indices = torch.arange(0, shape[0])
        flat_output[indices, ix] = flat_pred[indices, ix]

        # Suppression
        ix = ix.to(dtype=torch.float)
        y = ix / shape[-1]
        x = ix % shape[-1]
        mu = torch.stack([x, y], dim=1).float()
        g = neighborhoods(mu, shape[-1], shape[-2], sigma, gaussian=gaussian)
        supp_pred *= 1 - g.unsqueeze(1)

    # Make sure you always have at least one detection
    output[output < min(thresh, output.max())] = 0
    return output
