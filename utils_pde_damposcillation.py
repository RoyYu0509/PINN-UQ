from interface_pde import BasePDE

import torch
import torch.nn as nn
import math

class DampedOscillator1D(BasePDE):
    """1D damped oscillator ODE: u''(t) + 2*ζ*ω * u'(t) + ω^2 * u(t) = F(t), with initial conditions."""
    def __init__(self, zeta, omega, forcing_func, init_cond, domain, true_solution):
        """
        zeta: damping ratio ζ
        omega: natural frequency ω
        forcing_func: function F(t) (external force as function of t)
        init_cond: tuple (u(0), u'(0)) initial displacement and velocity at t=0
        """
        self.zeta = zeta
        self.omega = omega
        self.F = forcing_func
        self.u0, self.v0 = init_cond  # initial displacement and velocity
        self.x0, self.x1 = domain[0], domain[1]
        self.true_solution = true_solution

    def residual(self, model, coloc_pt_num):
        """ Sample collocation points in time domain [0, T], and compute residual at those points.
        Parameters:
            model: the neural network approximating u(t)
            coloc_pt_num: number of collocation points to sample
        Returns:
            residuals: tensor of shape [coloc_pt_num, 1]
        """
        # Uniformly sample collocation points (excluding t=0 for boundary condition)
        t = torch.linspace(self.x0, self.x1, coloc_pt_num).view(-1, 1)
        t = t.to(dtype=torch.float32)
        return (self._residual(model, t)**2).mean()

    def _residual(self, model, t):
        """Compute residual r(t) = u''(t) + 2ζω u'(t) + ω^2 u(t) - F(t)."""
        t = t.requires_grad_(True)
        u = model(t)            # u(t)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]   # first derivative
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]  # second derivative
        # Residual of the damped oscillator equation
        r = u_tt + 2 * self.zeta * self.omega * u_t + (self.omega**2) * u - self.F(t)
        return r

    def ic_loss(self, model):
        """Compute loss for initial conditions: (u(0)-u0)^2 + (u'(0)-v0)^2."""
        t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        u0_pred = model(t0)  # predicted u(0)
        u0_error = (u0_pred - self.u0)**2
        # Compute u'(0) via autograd
        u_t0 = torch.autograd.grad(u0_pred, t0, grad_outputs=torch.ones_like(u0_pred), create_graph=True)[0]
        v0_error = (u_t0 - self.v0)**2
        return (u0_error + v0_error).mean()


