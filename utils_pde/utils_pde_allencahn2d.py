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
        b_pts_n: int = 800,          # <── default number of boundary pts
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

    # interior = 1024

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
    # --- autograd-based forcing term (surgical tweak: device/dtype safety) ---
    def _forcing(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.requires_grad_(True)
        u = self.true_solution(xy)
        # ensure device/dtype match for autograd chain
        if u.device != xy.device or u.dtype != xy.dtype:
            u = u.to(device=xy.device, dtype=xy.dtype)

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

    # --- boundary sampler (surgical rewrite: even spacing, exact N, clear corners) ---
    def _build_boundary_pts(self, N: int) -> torch.Tensor:
        """Return exactly N evenly spaced boundary points with corners handled once."""
        N = int(max(0, N))
        if N == 0:
            return torch.empty(0, 2, dtype=torch.float32)

        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        Lx, Ly = (x1 - x0), (y1 - y0)

        # helper: interior linspace EXCLUDING endpoints
        def lin_exclude(a, b, m, *, device=None, dtype=None):
            if m <= 0:
                return torch.empty(0, device=device, dtype=dtype)
            t = torch.linspace(0.0, 1.0, m + 2, device=device, dtype=dtype)[1:-1]
            return a + (b - a) * t

        device = torch.device("cpu")
        dtype = torch.float32

        if N < 4:
            # place up to N corners in fixed order
            corners = torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                dtype=dtype, device=device)
            return corners[:N]

        # include corners once
        corners = torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                            dtype=dtype, device=device)
        remain = N - 4

        # distribute remaining interior edge points proportional to edge length
        weights = {"left": Ly, "right": Ly, "bottom": Lx, "top": Lx}
        W = sum(weights.values())
        raw = {e: remain * (w / W) for e, w in weights.items()}
        base = {e: int(math.floor(v)) for e, v in raw.items()}
        rem = remain - sum(base.values())
        # largest remainder rounding
        fracs = sorted(((raw[e] - base[e], e) for e in base.keys()), reverse=True)
        for k in range(rem):
            base[fracs[k][1]] += 1

        pts = [corners]
        # left edge interior (x=x0, y varying), exclude corners
        y = lin_exclude(y0, y1, base["left"], device=device, dtype=dtype)
        pts.append(torch.stack([torch.full_like(y, x0), y], dim=1))
        # right edge interior (x=x1)
        y = lin_exclude(y0, y1, base["right"], device=device, dtype=dtype)
        pts.append(torch.stack([torch.full_like(y, x1), y], dim=1))
        # bottom edge interior (y=y0)
        x = lin_exclude(x0, x1, base["bottom"], device=device, dtype=dtype)
        pts.append(torch.stack([x, torch.full_like(x, y0)], dim=1))
        # top edge interior (y=y1)
        x = lin_exclude(x0, x1, base["top"], device=device, dtype=dtype)
        pts.append(torch.stack([x, torch.full_like(x, y1)], dim=1))

        xy = torch.cat(pts, dim=0)
        # truncate in the unlikely event rounding overflowed
        return xy[:N]

    # --- boundary loss (surgical tweak: device/dtype & true_solution safety) ---
    def boundary_loss(self, model) -> torch.Tensor:
        """
        Dirichlet MSE on a deterministic set of boundary points (evenly spaced).
        Uses exactly self.b_pts_n points; corners are included once when b_pts_n >= 4.
        """
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        xy_bdy = self._build_boundary_pts(int(self.b_pts_n)).to(device=device, dtype=dtype)
        u_true = self.true_solution(xy_bdy)
        if u_true.device != xy_bdy.device or u_true.dtype != xy_bdy.dtype:
            u_true = u_true.to(device=xy_bdy.device, dtype=xy_bdy.dtype)

        return torch.mean((model(xy_bdy) - u_true) ** 2)

    # --- residual (surgical tweak: even interior grid + model dtype) ---
    def residual(self, model, coloc_pt_num: int) -> torch.Tensor:
        # strictly uniform interior grid (exclude boundary), matching model device/dtype
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        Lx, Ly = (self.x1 - self.x0), (self.y1 - self.y0)
        N = max(1, int(coloc_pt_num))
        b = math.sqrt(N / (Lx * Ly))
        Nx = max(1, math.ceil(b * Lx))
        Ny = max(1, math.ceil(b * Ly))

        xs = torch.linspace(self.x0, self.x1, Nx + 2, device=device, dtype=dtype)[1:-1]
        ys = torch.linspace(self.y0, self.y1, Ny + 2, device=device, dtype=dtype)[1:-1]
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
        xy.requires_grad_(True)

        return (self._residual(model, xy) ** 2).mean()

    # --- _residual (surgical tweak: shape normalize u before grads) ---
    def _residual(self, model, xy: torch.Tensor) -> torch.Tensor:
        u = model(xy)
        if u.ndim == 1:
            u = u.unsqueeze(-1)

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
    
    def solution_field_on_grid(
        self,
        n: int,
        *,
        include_boundary: bool = True,
        flatten: bool = True,
        source: str = "true",                 # "true" or "model"
        model: torch.nn.Module | None = None, # required if source=="model"
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        with_forcing: bool = False,
    ):
        """
        Generate an n×n uniform field of points and the corresponding solution values.

        Returns:
            If flatten=True:
                (xy, u) or (xy, u, f)
                - xy: (n*n, 2)
                - u:  (n*n, 1)
                - f:  (n*n, 1) if with_forcing
            If flatten=False:
                (X, Y, U) or (X, Y, U, F)
                - X, Y, U: (n, n)
                - F: (n, n) if with_forcing
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        # Resolve device sensibly
        if device is None and source == "model":
            if model is None:
                raise ValueError("When source='model', you must provide a model.")
            device = next(model.parameters()).device
        device = torch.device(device) if device is not None else torch.device("cpu")

        # Build grid (uses your earlier helper if present; otherwise inline)
        if hasattr(self, "uniform_grid_pts"):
            xy = self.uniform_grid_pts(
                n, include_boundary=include_boundary, flatten=True, device=device, dtype=dtype
            )
        else:
            # Inline grid (equivalent to uniform_grid_pts)
            if include_boundary:
                xs = torch.linspace(self.x0, self.x1, n, dtype=dtype, device=device)
                ys = torch.linspace(self.y0, self.y1, n, dtype=dtype, device=device)
            else:
                xs = torch.linspace(self.x0, self.x1, n + 2, dtype=dtype, device=device)[1:-1]
                ys = torch.linspace(self.y0, self.y1, n + 2, dtype=dtype, device=device)[1:-1]
            Xg, Yg = torch.meshgrid(xs, ys, indexing="ij")
            xy = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

        # Compute solution values
        if source == "true":
            with torch.no_grad():
                u = self.true_solution(xy)
        elif source == "model":
            if model is None:
                raise ValueError("When source='model', you must provide a model.")
            with torch.no_grad():
                u = model(xy)
        else:
            raise ValueError("source must be 'true' or 'model'")

        out = None
        if with_forcing:
            # Uses autograd internally; do NOT wrap in no_grad
            f = self._forcing(xy)

        if flatten:
            out = (xy, u) if not with_forcing else (xy, u, f)
        else:
            # Recover meshgrids for convenience
            if hasattr(self, "uniform_grid_pts"):
                Xg, Yg = self.uniform_grid_pts(
                    n, include_boundary=include_boundary, flatten=False, device=device, dtype=dtype
                )
            else:
                # Reuse the inline grids if they exist; otherwise reconstruct
                xs = xy[:, 0].reshape(n, n)  # consistent with meshgrid(indexing='ij')
                ys = xy[:, 1].reshape(n, n)
                Xg, Yg = xs, ys
            U = u.reshape(n, n)
            if with_forcing:
                F = f.reshape(n, n)
                out = (Xg, Yg, U, F)
            else:
                out = (Xg, Yg, U)

        return out
