from functools import partial

import torch
import torch.nn.functional as F

from fgmres import fgmres
from helmholtz import HelmholtzOperator

DOWN_KERNEL = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).unsqueeze(0).unsqueeze(0) * 1. / 16
UP_KERNEL = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).unsqueeze(0).unsqueeze(0) * 1. / 4


def jacobi(iterations: int, b: torch.Tensor, op: HelmholtzOperator, x: torch.Tensor = None,
           w: float = 0.8, shifted_laplacian=False) -> torch.Tensor:
    if x is None:
        x = torch.zeros_like(op.kappa, dtype=torch.cfloat)

    for _ in range(iterations):
        y = op(x, shifted_laplacian=shifted_laplacian).reshape_as(op.kappa)
        if shifted_laplacian:
            d = 4 / (op.h ** 2) - op.shifted_laplacian()
        else:
            d = 4 / (op.h ** 2) - op.helmholtz_operator()

        residual = b - y
        alpha = w / d

        x += alpha * residual

    return x


def down(matrix: torch.Tensor):
    return F.conv2d(matrix.unsqueeze(0).unsqueeze(0),
                    DOWN_KERNEL.to(device=matrix.device, dtype=matrix.dtype),
                    stride=2, padding=1).squeeze(0).squeeze(0)


def up(matrix: torch.Tensor):
    return F.conv_transpose2d(matrix.unsqueeze(0).unsqueeze(0),
                              UP_KERNEL.to(matrix.device, matrix.dtype), stride=2,
                              padding=1, output_padding=1).squeeze(0).squeeze(0)


def vcycle(levels: int, b: torch.Tensor, op: HelmholtzOperator, x: torch.Tensor = None,
           gmres_max_itr: int = 20, gmres_max_restart=1):
    if x is None:
        x = torch.zeros_like(op.kappa, dtype=torch.cfloat)

    if levels == 0:
        original_shape = x.shape
        sl_op = partial(op, shifted_laplacian=True)

        x = fgmres(sl_op, b.flatten(), max_iter=gmres_max_itr, x0=x.flatten(),
                   max_restarts=gmres_max_restart, rel_tol=1e-10)
        return x.solution.reshape(original_shape)
    else:
        x = jacobi(1, b, op, x, shifted_laplacian=True)
        residual = b - op(x).reshape_as(op.kappa)

        kappa_coarse = down(op.kappa)
        gamma_coarse = down(op.gamma)
        op_coarse = HelmholtzOperator(kappa_coarse, op.omega, gamma_coarse, op.h * 2)

        residual_coarse = down(residual)
        x_coarse = vcycle(levels - 1, residual_coarse,
                          op_coarse, None, gmres_max_itr)

        x = x + up(x_coarse)
        x = jacobi(1, b, op, x, shifted_laplacian=True)

    return x
