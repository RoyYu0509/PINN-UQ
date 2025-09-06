# utils_pde_helmholtz3d.py
# -------------------------------------------------------------------------- #
# 3-D Helmholtz equation on Ω = (x0,x1)×(y0,y1)×(z0,z1)
#
#     ∇²u(x,y,z) + k² u(x,y,z) = f(x,y,z)          in Ω
#                       u      = 0                 on ∂Ω      (Dirichlet)
#
# Analytic solution used for testing & synthetic data:
#     u*(x,y,z) = sin(πx)·sin(πy)·sin(πz)
# which gives
#     f(x,y,z) = (k² − 3π²)·sin(πx)·sin(πy)·sin(πz)
# -------------------------------------------------------------------------- #
import math
from typing import Callable, Tuple

import torch
from utils_pde.interface_pde import BasePDE


class Helmholtz3D(BasePDE):
    """
    3-D Helmholtz equation with zero Dirichlet BC on a rectangular box.

    Parameters
    ----------
    k : float
        Wave-number coefficient in ∇²u + k²u = f.
    domain : ((float,float), (float,float), (float,float))
        (x0,x1), (y0,y1), (z0,z1) describing Ω.
    true_solution : Callable[[Tensor], Tensor]
        Exact solution u*(x,y,z) for testing / synthetic data.
    b_pts_n : int
        points per face
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        k: float = math.pi,
        domain: Tuple[Tuple[float, float],
                      Tuple[float, float],
                      Tuple[float, float]] = ((0.0, 1.0),
                                              (0.0, 1.0),
                                              (0.0, 1.0)),
        true_solution: Callable[[torch.Tensor], torch.Tensor] = (
            lambda xyz: torch.sin(math.pi * xyz[..., 0:1])
            * torch.sin(math.pi * xyz[..., 1:2])
            * torch.sin(math.pi * xyz[..., 2:3])
        ),
        b_pts_n: int = 1024,
    ):
        self.k = k
        (self.x0, self.x1), (self.y0, self.y1), (self.z0, self.z1) = domain
        self.true_solution = true_solution
        self.b_pts_n = b_pts_n

    # interior points = 27000

    # ------------------------------------------------------------------ #
    # Forcing term f(x,y,z) matching the analytic solution
    # ------------------------------------------------------------------ #
    def _forcing(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute f for given (N,3) xyz points."""
        return (self.k ** 2 - 3 * math.pi ** 2) * (
            torch.sin(math.pi * xyz[..., 0:1])
            * torch.sin(math.pi * xyz[..., 1:2])
            * torch.sin(math.pi * xyz[..., 2:3])
        )

    # ------------------------------------------------------------------ #
    # Core residual  ∇²u + k²u − f    (no squaring / averaging here)
    # ------------------------------------------------------------------ #
    def _residual(self, model, xyz: torch.Tensor) -> torch.Tensor:
        xyz = xyz.requires_grad_(True)
        u = model(xyz)                                # u(x,y,z)
        grad_u = torch.autograd.grad(                 # ∇u
            u, xyz, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_x, u_y, u_z = grad_u.split(1, dim=1)

        u_xx = torch.autograd.grad(u_x, xyz,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xyz,
                                   grad_outputs=torch.ones_like(u_y),
                                   create_graph=True)[0][:, 1:2]
        u_zz = torch.autograd.grad(u_z, xyz,
                                   grad_outputs=torch.ones_like(u_z),
                                   create_graph=True)[0][:, 2:3]

        laplace_u = u_xx + u_yy + u_zz
        return laplace_u + self.k ** 2 * u - self._forcing(xyz)
    


    # ------------------------------------------------------------------ #
    # Collocation residual ‖·‖² averaged over N interior samples
    # ------------------------------------------------------------------ #
    def residual(self, model, coloc_pt_num: int) -> torch.Tensor:
        # strictly uniform interior grid (exclude boundary)
        device = next(model.parameters()).device
        Lx, Ly, Lz = (self.x1 - self.x0), (self.y1 - self.y0), (self.z1 - self.z0)

        # choose grid counts so spacing is uniform per axis and total ≥ target
        N = max(1, int(coloc_pt_num))
        a = (N / (Lx * Ly * Lz)) ** (1.0 / 3.0)
        Nx = max(1, math.ceil(a * Lx))
        Ny = max(1, math.ceil(a * Ly))
        Nz = max(1, math.ceil(a * Lz))

        xs = torch.linspace(self.x0, self.x1, Nx + 2, device=device, dtype=torch.float32)[1:-1]
        ys = torch.linspace(self.y0, self.y1, Ny + 2, device=device, dtype=torch.float32)[1:-1]
        zs = torch.linspace(self.z0, self.z1, Nz + 2, device=device, dtype=torch.float32)[1:-1]
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing="ij")
        xyz = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)

        return (self._residual(model, xyz) ** 2).mean()


    # ------------------------------------------------------------------ #
    # Dirichlet boundary loss  u|∂Ω = 0               (six cube faces)
    # ------------------------------------------------------------------ #
    def boundary_loss(self, model) -> torch.Tensor:
        p = next(model.parameters()); device, dtype = p.device, p.dtype
        n_face = int(self.b_pts_n)

        # choose nx*ny ~ n_face (deterministic)
        nx = max(1, int(round(math.sqrt(n_face))))
        ny = max(1, int(round(n_face / nx)))
        # exclude edges to avoid double-counting across faces
        xs = torch.linspace(self.x0, self.x1, nx + 2, device=device, dtype=dtype)[1:-1]
        ys = torch.linspace(self.y0, self.y1, ny + 2, device=device, dtype=dtype)[1:-1]
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        # Build 6 faces with interior face grids (no edges)
        faces = [
            torch.stack([torch.full((xy.size(0),), self.x0, device=device, dtype=dtype), xy[:,1], xy[:,0]], dim=1),  # x=x0
            torch.stack([torch.full((xy.size(0),), self.x1, device=device, dtype=dtype), xy[:,1], xy[:,0]], dim=1),  # x=x1
            torch.stack([xy[:,0], torch.full((xy.size(0),), self.y0, device=device, dtype=dtype), xy[:,1]], dim=1),  # y=y0
            torch.stack([xy[:,0], torch.full((xy.size(0),), self.y1, device=device, dtype=dtype), xy[:,1]], dim=1),  # y=y1
            torch.stack([xy[:,0], xy[:,1], torch.full((xy.size(0),), self.z0, device=device, dtype=dtype)], dim=1),  # z=z0
            torch.stack([xy[:,0], xy[:,1], torch.full((xy.size(0),), self.z1, device=device, dtype=dtype)], dim=1),  # z=z1
        ]
        xyz_bdry = torch.cat(faces, dim=0)  # total ≈ 6 * (nx*ny), deterministic and even

        u_pred = model(xyz_bdry)
        return (u_pred ** 2).mean()

    # ------------------------------------------------------------------ #
    # Synthetic observation sampler  (useful for PINN calibration / UQ)
    # ------------------------------------------------------------------ #
    def data_generation(
        self,
        size: int,
        noise: float = 0.0,
        true_solution: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int | None = None,
        device: torch.device | None = None,
    ):
        """
        Returns
        -------
        X : (size, 3) torch.Tensor  • random (x,y,z) points in Ω
        Y : (size, 1) torch.Tensor  • true_solution(X) with optional noise
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = torch.device("cpu") if device is None else device

        xs = torch.rand(size, 1, device=device) * (self.x1 - self.x0) + self.x0
        ys = torch.rand(size, 1, device=device) * (self.y1 - self.y0) + self.y0
        zs = torch.rand(size, 1, device=device) * (self.z1 - self.z0) + self.z0
        X = torch.cat([xs, ys, zs], dim=1)

        if true_solution is None:
            true_solution = self.true_solution
        Y_clean = true_solution(X).view(-1, 1)

        Y = Y_clean + noise * torch.randn_like(Y_clean) if noise > 0.0 else Y_clean
        return X.to(dtype=torch.float32), Y.to(dtype=torch.float32)
