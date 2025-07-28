# utils_pde_damposcillation.py  ── drop-in replacement (keeps original signature)
# -----------------------------------------------------------------------------#
# 1-D damped oscillator ODE
#     u''(t) + 2 ζ ω u'(t) + ω² u(t) = F(t)
# with initial conditions  u(t₀)=u₀,  u'(t₀)=v₀.
# -----------------------------------------------------------------------------#

import torch
import torch.nn as nn
from utils_pde.interface_pde import BasePDE


class DampedOscillator1D(BasePDE):
    """
    Damped-oscillator helper for (B-)PINNs.
    The API matches the original Poisson / Allen-Cahn PDE helpers:
        • `residual(model, N)` returns the mean-squared PDE residual
        • `ic_loss(model)`     returns the IC penalty
    """

    # ──────────────────────────────────────────────────────────────────────────
    # constructor  (same signature as before)
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        zeta,                     # damping ratio ζ
        omega,                    # natural frequency ω
        forcing_func,             # F(t) – must accept torch Tensor
        init_cond,                # (u₀, v₀)
        domain,                   # (t₀, t₁)
        true_solution,            # exact u*(t) – optional for testing
    ):
        # NB: we **don’t** call BasePDE.__init__ to stay perfectly aligned with
        #     your other PDE classes (they simply set the attributes below).
        self.zeta = float(zeta)
        self.omega = float(omega)
        self.F = forcing_func

        # initial displacement / velocity
        self.u0, self.v0 = init_cond

        # domain endpoints expected by BasePDE utilities (x0, x1)
        self.x0, self.x1 = domain
        # convenience aliases, in case later code uses t0/t1
        self.t0, self.t1 = self.x0, self.x1

        self.true_solution = true_solution

    # ──────────────────────────────────────────────────────────────────────────
    # private helper – move tensor to same device/dtype as reference
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _same_device(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return x.to(device=ref.device, dtype=ref.dtype)

    # ──────────────────────────────────────────────────────────────────────────
    # PDE residual ‖r(t)‖² over N random collocation points
    # ──────────────────────────────────────────────────────────────────────────
    def residual(self, model: nn.Module, coloc_pt_num: int) -> torch.Tensor:
        device = next(model.parameters()).device
        t = (self.x1 - self.x0) * torch.rand(
            coloc_pt_num, 1, device=device, dtype=torch.get_default_dtype()
        ) + self.x0
        t.requires_grad_(True)

        r = self._residual(model, t)
        return (r**2).mean()

    def _residual(self, model: nn.Module, t: torch.Tensor) -> torch.Tensor:
        """r(t) = u'' + 2ζω u' + ω² u − F(t)."""
        u = model(t)

        # u'(t)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        # u''(t)
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True
        )[0]

        zeta = self._same_device(torch.tensor(self.zeta), t)
        omega = self._same_device(torch.tensor(self.omega), t)

        return u_tt + 2.0 * zeta * omega * u_t + (omega**2) * u - self.F(t)

    # ──────────────────────────────────────────────────────────────────────────
    # initial-condition loss
    # ──────────────────────────────────────────────────────────────────────────
    def ic_loss(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        t0 = torch.tensor(
            [[self.x0]],
            dtype=torch.get_default_dtype(),
            device=device,
            requires_grad=True,
        )

        u0_pred = model(t0)
        u_t0 = torch.autograd.grad(
            u0_pred, t0, grad_outputs=torch.ones_like(u0_pred), create_graph=True
        )[0]

        u0_true = torch.as_tensor(self.u0, dtype=u0_pred.dtype, device=device)
        v0_true = torch.as_tensor(self.v0, dtype=u_t0.dtype, device=device)

        return ((u0_pred - u0_true) ** 2 + (u_t0 - v0_true) ** 2).mean()
