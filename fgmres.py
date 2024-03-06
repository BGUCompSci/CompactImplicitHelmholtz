import enum
from dataclasses import dataclass
from typing import Callable

import torch
import torch.linalg as la


@dataclass
class FGMRESResult:
    class ReturnCode(enum.Enum):
        TOLERANCE_ACHIEVED = 0
        MAX_ITER_REACHED = 1
        ZERO_RHS = 2

    solution: torch.Tensor
    ret_code: ReturnCode
    residual_norm: float
    num_iters: int
    residual_norms: list[float]
    intermediates: torch.Tensor = None


@dataclass
class FGMRESCache:
    V: torch.Tensor
    Z: torch.Tensor

    def __init__(self, device: torch.device, dtype: torch.dtype, n: int, k: int,
                 flexible: bool = False, num_rhs: int = 1):
        self.V = torch.zeros((n, k * num_rhs), device=device, dtype=dtype)

        if flexible:
            self.Z = torch.zeros((n, k * num_rhs), device=device, dtype=dtype)
        else:
            self.Z = torch.zeros((n, 0), device=device, dtype=dtype)


def fgmres(lhs_op: torch.Tensor | Callable[[torch.Tensor], torch.Tensor], rhs_vec: torch.Tensor,
           max_restarts: int = None,
           rel_tol: float = 1e-2, max_iter: int = 100, precond: Callable[[torch.Tensor], torch.Tensor] = None,
           x0: torch.Tensor = None, flexible: bool = False,
           save_intermediate: bool = False, cache: FGMRESCache = None,
           verbose: bool = False) -> FGMRESResult:
    """Solves the linear system Ax = b using the Flexible Generalized Minimal Residual (FGMRES) method.

    Args:
        lhs_op: The left-hand side operator of the linear system. Can be a matrix or a callable that
            takes a vector and returns the result of the matrix-vector multiplication with the matrix A.
        rhs_vec: The right-hand side vector of the linear system.
        max_restarts: The maximum number of Krylov restarts. If None, it is set to n - 1.
        rel_tol: The relative tolerance for the residual norm. The algorithm stops if the residual norm
            is less than rel_tol * rhs_vec.norm().
        max_iter: The maximum number of iterations.
        precond: The preconditioner. If None, no preconditioning is used.
        x0: The initial guess. If None, it is set to zero.
        flexible: If True, the flexible version of FGMRES is used.
        save_intermediate: If True, the intermediate solutions are saved and returned. Otherwise, only
            the final solution is returned and the intermediates field of the result is None.
        cache: A cache object that can be used to reuse the Krylov basis vectors and the preconditioned
            Krylov basis vectors. If None, a new cache object is created.
        verbose: If True, the residual norm is printed at each iteration.

    Returns:
        A FGMRESResult object containing:
            solution: The solution of the linear system.
            ret_code: The return code of the algorithm. Can be one of:
                TOLERANCE_ACHIEVED: The algorithm converged to the desired tolerance.
                MAX_ITER_REACHED: The algorithm reached the maximum number of iterations.
                ZERO_RHS: The right-hand side vector is zero.
            residual_norm: The residual norm of the final iterate.
            num_iters: The number of iterations performed.
            residual_norms: A list containing the residual norms at each iteration.
            intermediates: A tensor containing the intermediate solutions.
                If save_intermediate is False, this field is None."""
    n = rhs_vec.shape[0]
    device = rhs_vec.device
    dtype = rhs_vec.dtype
    eps = torch.finfo(dtype).eps
    if isinstance(lhs_op, torch.Tensor):
        assert lhs_op.dtype == dtype, 'lhs_op and rhs_vec must have the same dtype'
        assert lhs_op.device == device, 'lhs_op and rhs_vec must be on the same device'

        def mv(v):
            return torch.mv(lhs_op, v)
    elif callable(lhs_op):
        mv = lhs_op
    else:
        raise TypeError('lhs_op must be a matrix or a callable')

    if (len(rhs_vec.shape)) == 2:
        num_rhs = rhs_vec.shape[1]
    else:
        num_rhs = 1

    if cache is None:
        cache = FGMRESCache(device, dtype, n, max_restarts, flexible, num_rhs)
    else:
        assert cache.V.shape[0] == n, \
            'The cache Krylov matrix has the wrong size, it should be n x k'
        assert cache.V.shape[1] % max_restarts == 0, \
            'The cache Krylov matrix has the wrong size, ' \
            'the number of columns should be a multiple of max_restarts'
        assert cache.Z.shape[0] == n, \
            'The cache preconditioned Krylov matrix has the wrong size, it should be n x k'

    if precond is None:
        def precond(x): return x

    if rhs_vec.norm() == 0.0:
        return FGMRESResult(torch.zeros(n, dtype=dtype),
                            FGMRESResult.ReturnCode.ZERO_RHS,
                            0.0, 0, [0.0])

    if x0 is None:
        x0 = torch.zeros(n, dtype=dtype, device=device)
        r = rhs_vec.clone()
    elif torch.norm(x0) < eps:
        r = rhs_vec.clone()
    else:
        r = rhs_vec.clone() - mv(x0)

    if save_intermediate:
        intermeds = torch.zeros(n, max_iter, dtype=dtype, device=device)
    else:
        intermeds = None

    beta = r.norm()
    rnorm0 = rhs_vec.norm()

    if rnorm0 == 0.0:
        rnorm0 = 1.0

    rel_err = r.norm() / rnorm0
    if rel_err < rel_tol:
        return FGMRESResult(x0, FGMRESResult.ReturnCode.TOLERANCE_ACHIEVED,
                            rel_err, 0, [rel_err])

    if verbose:
        print(f'Initial residual norm: {rnorm0}, relative error: {rel_err}')

    max_restarts = min(max_restarts, n - 1)
    H = torch.zeros((max_restarts + 1, max_restarts), dtype=dtype, device=device)
    xi = torch.zeros(max_restarts + 1, dtype=dtype, device=device)
    t = torch.zeros(max_restarts, dtype=dtype, device=device)

    status = FGMRESResult.ReturnCode.MAX_ITER_REACHED
    res_vec = []
    w = torch.zeros(0, dtype=dtype, device=device)
    num_iters = 0
    for i in range(max_iter):
        xi[0] = beta
        r = r / beta
        H[:] = 0.0
        t[:] = 0.0

        for j in range(max_restarts):
            if j == 0:
                cache.V[:, j] = r
                z = precond(r)
            else:
                cache.V[:, j] = w
                z = precond(w)

            if flexible:
                cache.Z[:, j] = z

            num_iters += 1

            w = mv(z)

            # Modified Gram-Schmidt process (MGS):
            # for i in range(1, j + 1):
            #     H[i, j] = torch.dot(vec(V[:, i]), w)
            #     w = w - H[i, j] * vec(V[:, i])

            # Gram-Schmidt
            # Much faster than MGS even though zeros are multiplied, does relatively well
            t = cache.V.H @ w
            t[j + 1:] = 0.0
            H[:max_restarts, j] = t
            w = w - cache.V @ t

            beta = w.norm()
            w = w / beta
            H[j + 1, j] = beta

            # the following 2 lines are equivalent to the 2 next
            # y = H[1:j + 1, 1:j] \ xi[1:j + 1]
            # rel_err = norm(H[1:j + 1, 1:j] * y - xi[1:j + 1]) / rnorm0
            Q = la.qr(H[:j + 2, :j + 1], mode='complete').Q
            rel_err = torch.abs(Q[0, -1] * xi[0]) / rnorm0
            res_vec.append(rel_err.item())

            if verbose:
                print(f'Iteration {i + 1}, restart {j + 1}, '
                      f'residual norm: {rel_err * rnorm0}, relative error: {rel_err}')

            if rel_err <= rel_tol:
                if flexible:
                    cache.Z[:, j + 1:] = 0.0
                else:
                    cache.V[:, j + 1:] = 0.0

                status = FGMRESResult.ReturnCode.TOLERANCE_ACHIEVED
                break

        # Solve the least squares problem to get the correction to x0
        y = la.pinv(H) @ xi

        if flexible:
            # This is the correction that corresponds to the residual
            w = cache.Z @ y
        else:
            w = cache.V @ y
            z = precond(w)
            w[:] = z

        x0 = x0 + w

        if save_intermediate:
            intermeds[:, i] = x0

        if rel_err <= rel_tol:
            status = FGMRESResult.ReturnCode.TOLERANCE_ACHIEVED
            break

        # Restart if we haven't converged
        if i < max_iter:
            r = rhs_vec.clone() - mv(x0)
            beta = r.norm()

    if verbose:
        print(f'Final residual norm: {r.norm()}, relative error: {r.norm() / rnorm0}')

    if save_intermediate:
        return FGMRESResult(x0, status, res_vec[-1], num_iters, res_vec, intermeds[:, :i])
    else:
        return FGMRESResult(x0, status, res_vec[-1], num_iters, res_vec)
