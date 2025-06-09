from abc import ABC, abstractmethod
import torch

class BaseUQPINN(ABC):
    """
    Abstract base class for uncertainty quantification in Physics-Informed Neural Networks (PINNs).

    All UQ methods should subclass this and implement the core methods for training and prediction.
    """

    def __init__(self, model: torch.nn.Module, pde, x_collocation: torch.Tensor,
                 x_boundary: torch.Tensor = None, boundary_values=None):
        """
        Parameters:
            model: the neural network (can be deterministic or Bayesian).
            pde: instance of a PDE class with .residual(model, x) and .boundary_loss(model) methods.
            x_collocation: interior points for computing physics residual loss.
            x_boundary: boundary/initial condition points (optional).
            boundary_values: actual boundary values, optional depending on PDE class.
        """
        self.model = model
        self.pde = pde
        self.x_collocation = x_collocation
        self.x_boundary = x_boundary
        self.boundary_values = boundary_values

    @abstractmethod
    def train(self, num_epochs: int = 10000, lr: float = 1e-3, print_every: int = 500):
        """
        Train the model using the appropriate UQ strategy.
        """
        pass

    @abstractmethod
    def predict(self, x_test: torch.Tensor, **kwargs):
        """
        Return predictive mean and uncertainty estimates (e.g., variance or quantiles).
        Returns:
            mean: Tensor of shape [batch_size, output_dim]
            uncertainty: Tensor of shape [batch_size, output_dim] (e.g., std, interval width, etc.)
        """
        pass
