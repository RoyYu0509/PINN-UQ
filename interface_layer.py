from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseLayer(ABC, nn.Module):
    """
    Abstract base class for any layer in a PINN (Bayesian or Deterministic).
    All derived layers must implement forward() and kl_divergence().
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for this layer.
        """
        pass

    def kl_divergence(self) -> torch.Tensor:
        """
        Return the KL divergence of this layer.
        Default is zero (for deterministic layers).
        Override in Bayesian layers.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)
