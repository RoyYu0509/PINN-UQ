from interface_pde import BasePDE

import torch
import torch.nn as nn
import math
# utils_pde_poisson2d.py
# ---------------------------------------------------------------------
# 2-D Poisson equation on Ω = (0,1) × (0,1)
#   Δu(x,y) + 2π² sin(πx) sin(πy) = 0,
# with homogeneous Dirichlet boundaries u = 0 on ∂Ω.
# The analytic solution is  u*(x,y) = sin(πx)·sin(πy).
# ---------------------------------------------------------------------
from interface_pde import BasePDE

import torch
import math


class Poisson2D(BasePDE):
    """2-D Poisson equation with zero Dirichlet BC on the unit square."""

    def __init__(self,
                 domain=((0.0, 1.0), (0.0, 1.0)),
                 true_solution=lambda xy: torch.sin(math.pi * xy[..., 0:1]) *
                                          torch.sin(math.pi * xy[..., 1:2])):
        """
        domain: ((x0, x1), (y0, y1)) describing Ω.
        true_solution: callable for the exact u(x, y); useful for UQ metrics.
        """
        self.x0, self.x1 = domain[0]
        self.y0, self.y1 = domain[1]
        self.true_solution = true_solution  # keep same attribute name convention

    # ------------------------------------------------------------------
    # Forcing term f(x,y) chosen so that u* = sin(πx) sin(πy) is exact.
    # ------------------------------------------------------------------
    def _forcing(self, xy):
        x = xy[..., 0:1]
        y = xy[..., 1:2]
        return 2 * (math.pi ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)

    # ------------------------------------------------------------------
    # PDE residual ‖ Δu + f ‖² on a set of interior collocation points
    # ------------------------------------------------------------------
    def residual(self, model, coloc_pt_num):
        # Uniform interior sampling
        x = torch.rand(coloc_pt_num, 1) * (self.x1 - self.x0) + self.x0
        y = torch.rand(coloc_pt_num, 1) * (self.y1 - self.y0) + self.y0
        xy = torch.cat([x, y], dim=1).to(dtype=torch.float32)
        return (self._residual(model, xy) ** 2).mean()

    def _residual(self, model, xy):
        xy = xy.requires_grad_(True)
        u = model(xy)  # u(x,y)
        grad_u = torch.autograd.grad(
            u, xy, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]  # ∇u = (u_x, u_y)

        u_x = grad_u[:, 0:1]
        u_y = grad_u[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]  # u_xx
        u_yy = torch.autograd.grad(
            u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True
        )[0][:, 1:2]  # u_yy

        return u_xx + u_yy + self._forcing(xy)  # Δu + f(x,y)

    # ------------------------------------------------------------------
    # Dirichlet boundary loss (u ≡ 0 on ∂Ω)
    # ------------------------------------------------------------------
    def boundary_loss(self, model, n_b=500):
        # two random sets of points along each pair of opposing edges
        x = torch.rand(n_b, 1) * (self.x1 - self.x0) + self.x0
        y = torch.rand(n_b, 1) * (self.y1 - self.y0) + self.y0

        left = torch.cat([torch.full_like(y, self.x0), y], dim=1)
        right = torch.cat([torch.full_like(y, self.x1), y], dim=1)
        bottom = torch.cat([x, torch.full_like(x, self.y0)], dim=1)
        top = torch.cat([x, torch.full_like(x, self.y1)], dim=1)

        xy_bdry = torch.cat([left, right, bottom, top], dim=0).to(dtype=torch.float32)
        u_pred = model(xy_bdry)
        return (u_pred ** 2).mean()

    # ------------------------------------------------------------------
    # Generate synthetic observations in 2-D
    # ------------------------------------------------------------------
    def data_generation(self, size, noise=0.0, true_solution=None,
                    seed=None, device=None):
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device("cpu") if device is None else device

        # 1️⃣ sample
        xs = (self.x1 - self.x0) * torch.rand(size, 1, device=device) + self.x0
        ys = (self.y1 - self.y0) * torch.rand(size, 1, device=device) + self.y0
        X_train = torch.cat([xs, ys], dim=1)           # (size,2)

        # 2️⃣ evaluate truth
        if true_solution is None:
            if hasattr(self, "true_solution") and callable(self.true_solution):
                true_solution = self.true_solution
            else:
                raise ValueError("Provide `true_solution` or set `self.true_solution`.")
        Y_clean = true_solution(X_train).view(-1, 1).to(device)  # (size,1)

        # 3️⃣ optional noise
        if noise > 0.0:
            Y_train = Y_clean + noise * torch.randn_like(Y_clean)
        else:
            Y_train = Y_clean.clone()

        return X_train, Y_train