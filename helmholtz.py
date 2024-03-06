import torch
from torch import nn

from multigrid import vcycle

LAPLACE_KERNEL = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)


class HelmholtzOperator(nn.Module):
    def __init__(self, kappa: torch.Tensor, omega: float, gamma: torch.Tensor, h: float) -> None:
        super().__init__()

        self.kappa = kappa
        self.omega = omega
        self.gamma = gamma
        self.h = h

        self.device = kappa.device
        self.laplace_kernel = LAPLACE_KERNEL.to(self.device, kappa.dtype) * (1. / (self.h ** 2))

    def helmholtz_operator(self):
        return ((self.kappa ** 2) * self.omega) * (self.omega - self.gamma * 1j)

    def shifted_laplacian(self, alpha=0.5):
        return ((self.kappa ** 2) * self.omega) * ((self.omega - self.gamma * 1j) - (1j * self.omega * alpha))

    def forward(self, x, shifted_laplacian: bool = False):
        original_shape = x.shape

        if x.dim() == 1:
            x = x.reshape_as(self.kappa)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        convolution = torch.nn.functional.conv2d(x, self.laplace_kernel, padding=1)

        if shifted_laplacian:
            return (convolution - self.shifted_laplacian() * x).reshape(original_shape)
        return (convolution - self.helmholtz_operator() * x).reshape(original_shape)


def absorbing_layer(gamma, pad, amp, neumann_first_dim=False):
    device = gamma.device
    n = gamma.shape
    b_bwd1 = ((torch.arange(pad[0], 0, -1, device=device) ** 2) / (pad[0] ** 2)).reshape(-1, 1)
    b_bwd2 = ((torch.arange(pad[1], 0, -1, device=device) ** 2) / (pad[1] ** 2)).reshape(-1, 1)

    b_fwd1 = ((torch.arange(1, pad[0] + 1, device=device) ** 2) / (pad[0] ** 2)).reshape(-1, 1)
    b_fwd2 = ((torch.arange(1, pad[1] + 1, device=device) ** 2) / (pad[1] ** 2)).reshape(-1, 1)

    I1 = torch.arange(n[0] - pad[0], n[0], device=device)
    I2 = torch.arange(n[1] - pad[1], n[1], device=device)

    if not neumann_first_dim:
        gamma[:, :pad[1]] += torch.ones(n[0], 1, device=device) @ b_bwd2.T * amp
        gamma[:pad[0], :pad[1]] -= (b_bwd1 @ b_bwd2.T) * amp
        gamma[I1, :pad[1]] -= b_fwd1 @ b_bwd2.T * amp

    gamma[:, I2] += (torch.ones(n[0], 1, device=device) @ b_fwd2.T) * amp
    gamma[:pad[0], :] += (b_bwd1 @ torch.ones(1, n[1], device=device)) * amp
    gamma[I1, :] += (b_fwd1 @ torch.ones(1, n[1], device=device)) * amp
    gamma[:pad[0], I2] -= (b_bwd1 @ b_fwd2.T) * amp
    gamma[I1[0]:I1[-1] + 1, I2[0]:I2[-1] + 1] -= (b_fwd1 @ b_fwd2.T) * amp

    return gamma


def nn_precond(op: HelmholtzOperator, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        height, width = [*op.kappa.shape]

        # Split residual from complex form to channel representation
        r = x.reshape(height, width)
        R_split = torch.stack((r.real, r.imag))

        kappa = op.kappa.unsqueeze(0) ** 2
        network_input = torch.concat((R_split, kappa)).unsqueeze(0)

        e = model(network_input).squeeze(0)
        e = torch.complex(e[0,], e[1,]).reshape(height, width) * (op.h ** 2)
        e = vcycle(levels=3, b=r, op=op, x=e.detach())

        return e.flatten()
