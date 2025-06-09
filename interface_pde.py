from abc import ABC, abstractmethod
import torch

class BasePDE(ABC):
    """
    Abstract base class for PDE problems.
    All PDE classes must implement the residual() and boundary_loss() methods.
    This is to separate the calculation of the residue loss and boundary loss
    from the NN model.
    """

    @abstractmethod
    def residual(self, model: torch.nn.Module, x_coloc: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual of the PDE at input points x.
        Parameters:
            model: the neural network approximating the PDE solution
            x_coloc: input tensor with shape [N, 1] where N is number of points
        Returns:
            Tensor of residuals at each point
        """
        pass

    @abstractmethod
    def boundary_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute the loss associated with boundary/initial conditions.
        Parameters:
            model: the neural network approximating the PDE solution
        Returns:
            Scalar tensor representing the boundary/initial condition loss
        """
        pass
