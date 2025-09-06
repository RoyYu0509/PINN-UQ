from utils_pde.interface_pde import *

# We define classes for specific PDEs to compute residuals and boundary losses.
class Poisson1D(BasePDE):
    """1D Poisson equation: u''(x) = f(x), on domain [x0, x1] with Dirichlet boundary conditions."""
    def __init__(self, f_func, bc_values, domain, true_solution):
        """
        f_func: a function f(x) defining the source term.
        domain: tuple (x0, x1) specifying domain interval.
        bc_values: tuple (u(x0), u(x1)) Dirichlet boundary conditions.
        """
        self.f = f_func
        self.x0, self.x1 = domain
        self.u0, self.u1 = bc_values  # boundary condition values
        self.true_solution = true_solution

    def residual(self, model, coloc_pt_num):
        """
        Sample collocation points in time domain [0, T], and compute residual at those points.
        Parameters:
            model: the neural network approximating u(t)
            coloc_pt_num: number of collocation points to sample
        Returns:
            residuals: tensor of shape [coloc_pt_num, 1]
        """
        # Uniformly sample collocation points (excluding t=0 for boundary condition)
        # t = torch.linspace(self.x0, self.x1, coloc_pt_num).view(-1, 1)
        x = torch.linspace(self.x0, self.x1, steps=coloc_pt_num+2)[1:-1].unsqueeze(-1)
        x = x.to(dtype=torch.float32)
        return (self._residual(model, x) ** 2).mean()

    def _residual(self, model, x):
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

