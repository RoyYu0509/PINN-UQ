from interface_pde import *

# We define classes for specific PDEs to compute residuals and boundary losses.
class Poisson1D(BasePDE):
    """1D Poisson equation: u''(x) = f(x), on domain [x0, x1] with Dirichlet boundary conditions."""
    def __init__(self, f_func, domain, bc_values):
        """
        f_func: a function f(x) defining the source term.
        domain: tuple (x0, x1) specifying domain interval.
        bc_values: tuple (u(x0), u(x1)) Dirichlet boundary conditions.
        """
        self.f = f_func
        self.x0, self.x1 = domain
        self.u0, self.u1 = bc_values  # boundary condition values

    def residual(self, model, x):
        """Compute the PDE residual r(x) = u''(x) - f(x)."""
        # Ensure x requires grad for autograd (for computing derivatives)
        x = x.requires_grad_(True)
        u = model(x)            # forward pass to get the solution u(x)
        # First derivative u_x
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Second derivative u_xx
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        # Compute residual: u''(x) - f(x)
        r = u_xx - self.f(x)
        return r

    def boundary_loss(self, model):
        """Compute MSE loss for boundary conditions: (u(x0)-u0)^2 + (u(x1)-u1)^2."""
        # Evaluate model at boundary points (as 1D, input must be tensor of shape [N,1])
        x0_tensor = torch.tensor([[self.x0]], dtype=torch.float32)
        x1_tensor = torch.tensor([[self.x1]], dtype=torch.float32)
        u0_pred = model(x0_tensor)
        u1_pred = model(x1_tensor)
        # Mean squared error on boundary constraints
        loss_bc = (u0_pred - self.u0)**2 + (u1_pred - self.u1)**2
        return loss_bc.mean()  # mean (if multiple points, though here just two points)
