from abc import ABC, abstractmethod
import torch
from typing import Callable

class BasePDE(ABC):
    """
    Abstract base class for PDE problems.
    All PDE classes must implement the residual() and boundary_loss() methods.
    This is to separate the calculation of the residue loss and boundary loss
    from the NN model.
    """

    def __init__(self, domain: tuple, true_solution: Callable):
        """
        Parameters:
            true_solution: the function that maps the input to the true solution described
            by the underlying pde
        """
        # Unpack the domain
        # Save the true solution expression

    def data_generation(
        self,
        size: int,
        noise: float = 0.0,
        true_solution: Callable = None,
        seed: int = None,
        device: torch.device = None
    ):
        """
        Generate noisy observations (X_train, Y_train).

        Parameters
        ----------
        size : int
            Number of training points.
        noise : float, default 0.0
            Standard deviation σ of i.i.d. Gaussian noise ε~𝒩(0,σ²) added to the
            noiseless target.
        true_solution : callable, optional
            Function u*(x) that returns the exact solution at x (torch tensor in, out).
            If not provided, the method looks for self.true_solution.  Raises an
            error if neither is available.
        seed : int, optional
            Random-seed for reproducibility.
        device : torch.device, optional
            Put the tensors on this device (defaults to CPU).

        Returns
        -------
        X_train : torch.Tensor  shape [size, 1]
        Y_train : torch.Tensor  shape [size, 1]   (u*(x) + ε)
        """
        if seed is not None:
            torch.manual_seed(seed)

        if device is None:
            device = torch.device("cpu")

        # 1. Sample x uniformly from the domain
        X_train = (self.x1 - self.x0) * torch.rand(size, 1, device=device) + self.x0
        X_train = X_train.float()

        # 2. Obtain noiseless ground-truth u*(x)
        if true_solution is None:
            if hasattr(self, "true_solution") and callable(self.true_solution):
                true_solution = self.true_solution
            else:
                raise ValueError(
                    "You must supply a `true_solution` function "
                    "or set `self.true_solution` before calling data_generation."
                )
        Y_clean = true_solution(X_train)

        # 3. Add Gaussian noise ε ~ 𝒩(0, σ²)
        if noise > 0.0:
            Y_train = Y_clean + noise * torch.randn_like(Y_clean)
        else:
            Y_train = Y_clean.clone()

        return X_train, Y_train

#
#     @abstractmethod
#     def residual(self, model: torch.nn.Module, x_coloc: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the residual of the PDE at input points x.
#         Parameters:
#             model: the neural network approximating the PDE solution
#             x_coloc: input tensor with shape [N, 1] where N is number of points
#         Returns:
#             Tensor of residuals at each point
#         """
#         pass
#
#
#     def boundary_loss(self, model: torch.nn.Module) -> torch.Tensor:
#         """
#         Compute the loss associated with boundary/initial conditions.
#         Parameters:
#             model: the neural network approximating the PDE solution
#         Returns:
#             Scalar tensor representing the boundary/initial condition loss
#         """
#         pass

