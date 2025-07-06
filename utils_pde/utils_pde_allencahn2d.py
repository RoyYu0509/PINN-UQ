# utils_pde_allencahn2d.py  (drop-in replacement)
# -----------------------------------------------------------------------------#
# Allen-Cahn 2-D   —  now with built-in boundary sampler like Poisson2D
# -----------------------------------------------------------------------------#
import math
from typing import Callable, Tuple, Optional

import numpy as np
import torch

from utils_pde.interface_pde import BasePDE


class AllenCahn2D():
    """
    λ (∂²ₓu + ∂²ᵧu) + u(u²−1) = f  in Ω,      u = u_exact  on ∂Ω   (Dirichlet)

    *Single* attribute `true_solution` drives everything (forcing, data, etc.).
    Boundary handling now mirrors your Poisson2D utility:

        • `b_pts_n` controls how many boundary points are used per call.
        • `boundary_loss(model, n_b=...)` internally samples those points.
        • `boundary_mask(xy)` is provided for any custom filtering.
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        lam: float = 0.01,
        domain: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, 1.0),
                                                                   (-1.0, 1.0)),
        true_solution: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        b_pts_n: int = 500,          # <── default number of boundary pts
    ):
        self.lam = lam
        self.domain = domain
        self.x0, self.x1 = domain[0]
        self.y0, self.y1 = domain[1]
        self.b_pts_n = b_pts_n

        if true_solution is None:
            true_solution = (
                lambda xy: torch.sin(math.pi * xy[..., 0:1]) *
                           torch.sin(math.pi * xy[..., 1:2])
            )
        self.true_solution = true_solution

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _sample_interior(self, N: int) -> torch.Tensor:
        x = torch.rand(N, 1) * (self.x1 - self.x0) + self.x0
        y = torch.rand(N, 1) * (self.y1 - self.y0) + self.y0
        return torch.cat([x, y], dim=1).to(torch.float32)

    # ------------------------------------------------------------------ #
    # autograd-based forcing term
    # ------------------------------------------------------------------ #
    def _forcing(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.requires_grad_(True)
        u = self.true_solution(xy)
        grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grads[:, 0:1], xy,
                                   torch.ones_like(grads[:, 0:1]),
                                   create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(grads[:, 1:2], xy,
                                   torch.ones_like(grads[:, 1:2]),
                                   create_graph=True)[0][:, 1:2]
        lap = u_xx + u_yy
        nonlinear = u ** 3 - u
        return self.lam * lap + nonlinear

    # ------------------------------------------------------------------ #
    # deterministic, evenly–spaced boundary set
    # ------------------------------------------------------------------ #
    def _build_boundary_pts(self, N: int) -> torch.Tensor:
        """Return N points distributed (almost) equally on the 4 edges."""
        N = int(N)
        per_edge = [N // 4] * 4
        for k in range(N % 4):  # distribute remainder deterministically
            per_edge[k] += 1

        # helper to linspace without duplicating corners
        def edge_linspace(a, b, m, include_first=True, include_last=True):
            if m == 0:
                return torch.empty(0)
            if m == 1:
                return torch.tensor([a if include_first else b])
            t = torch.linspace(0.0, 1.0, m)
            if not include_first:
                t = t[1:]
            if not include_last:
                t = t[:-1]
            return a + (b - a) * t

        # Build edges
        pts = []

        # left  (x = x0, varying y)  – include both corners
        y_left = edge_linspace(self.y0, self.y1, per_edge[0])
        pts.append(torch.stack([torch.full_like(y_left, self.x0), y_left], dim=1))

        # right (x = x1)
        y_right = edge_linspace(self.y0, self.y1, per_edge[1],
                                include_first=False, include_last=False)
        pts.append(torch.stack([torch.full_like(y_right, self.x1), y_right], dim=1))

        # bottom (y = y0)
        x_bot = edge_linspace(self.x0, self.x1, per_edge[2], include_first=False)
        pts.append(torch.stack([x_bot, torch.full_like(x_bot, self.y0)], dim=1))

        # top (y = y1)
        x_top = edge_linspace(self.x0, self.x1, per_edge[3], include_first=False,
                              include_last=False)
        pts.append(torch.stack([x_top, torch.full_like(x_top, self.y1)], dim=1))

        return torch.cat(pts, dim=0).to(torch.float32)[:N]  # exact N points

    # ------------------------------------------------------------------ #
    # boundary loss   (now deterministic)
    # ------------------------------------------------------------------ #
    def boundary_loss(self, model) -> torch.Tensor:
        """
        Dirichlet MSE on a deterministic set of boundary points.
        If n_b is given and ≠ self.b_pts_n, a temporary deterministic set
        of that size is generated; otherwise the cached one is used.
        """
        xy_bdy = self._build_boundary_pts(int(self.b_pts_n))
        # ensure device match
        xy_bdy = xy_bdy.to(next(model.parameters()).device, non_blocking=True)
        return torch.mean((model(xy_bdy) - self.true_solution(xy_bdy)) ** 2)

    # ------------------------------------------------------------------ #
    # PDE residual   ‖λΔu + u(u²−1) − f‖²
    # ------------------------------------------------------------------ #
    def residual(self, model, coloc_pt_num: int) -> torch.Tensor:
        xy = self._sample_interior(coloc_pt_num)
        return (self._residual(model, xy) ** 2).mean()

    def _residual(self, model, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.requires_grad_(True)
        u = model(xy)
        grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(grads[:, 0:1], xy,
                                   torch.ones_like(grads[:, 0:1]),
                                   create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(grads[:, 1:2], xy,
                                   torch.ones_like(grads[:, 1:2]),
                                   create_graph=True)[0][:, 1:2]
        lap = u_xx + u_yy
        nonlinear = u ** 3 - u
        return self.lam * lap + nonlinear - self._forcing(xy)

    # ------------------------------------------------------------------ #
    # synthetic data generation
    # ------------------------------------------------------------------ #
    def data_generation(
        self,
        N: int,
        noise_std: float = 0.0,
        as_tensor: bool = True,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)
        xy = self._sample_interior(N).cpu().numpy()
        with torch.no_grad():
            u = self.true_solution(torch.tensor(xy, dtype=torch.float32)).cpu().numpy()
        if noise_std > 0:
            u += np.random.normal(0.0, noise_std, size=u.shape)

        if as_tensor:
            return (torch.tensor(xy, dtype=torch.float32),
                    torch.tensor(u,  dtype=torch.float32))
        return xy, u
