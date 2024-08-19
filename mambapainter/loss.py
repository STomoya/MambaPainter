from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.cuda.amp import GradScaler


def ns_real_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-logits).mean()


def ns_fake_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits).mean()


def ns_d_loss(real_prob: torch.Tensor, fake_prob: torch.Tensor) -> torch.Tensor:
    """Calc discriminator loss.

    Args:
    ----
        real_prob (torch.Tensor): D output of real inputs
        fake_prob (torch.Tensor): D output of fake inputs

    Returns:
    -------
        torch.Tensor: The loss

    """
    rl = ns_real_loss(real_prob)
    fl = ns_fake_loss(fake_prob)
    loss = rl + fl
    return loss


def ns_g_loss(fake_prob: torch.Tensor) -> torch.Tensor:
    """Calc generator loss.

    Args:
    ----
        fake_prob (torch.Tensor): D output of fake inputs.

    Returns:
    -------
        torch.Tensor: The loss

    """
    return ns_real_loss(fake_prob)


@torch.cuda.amp.autocast(enabled=False)
def _calc_grad(outputs: torch.Tensor, inputs: torch.Tensor, scaler: GradScaler) -> torch.Tensor:
    """Calculate gradients with AMP support.

    Args:
    ----
        outputs (torch.Tensor): Output tensor from a model
        inputs (torch.Tensor): Input tensor to a model
        scaler (Optional[GradScaler], optional): GradScaler object if using AMP. Defaults to None.

    Returns:
    -------
        torch.Tensor: The gradients of the input.

    """
    if isinstance(scaler, GradScaler):
        outputs = scaler.scale(outputs)
    ones = torch.ones(outputs.size(), device=outputs.device)
    gradients = grad(
        outputs=outputs, inputs=inputs, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    if isinstance(scaler, GradScaler):
        gradients = gradients / scaler.get_scale()
    return gradients


def prepare_input(real: torch.Tensor) -> torch.Tensor:
    return real.requires_grad_(True)


def r1_regularization(outputs: torch.Tensor, inputs: torch.Tensor, scaler: GradScaler):
    gradients = _calc_grad(outputs, inputs, scaler)
    gradients = gradients.reshape(gradients.size(0), -1)
    penalty = gradients.norm(2, dim=1).pow(2).mean() / 2.0
    return penalty
