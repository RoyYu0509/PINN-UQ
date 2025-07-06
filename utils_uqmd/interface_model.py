from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BasePINNModel(ABC, nn.Module):
    """
    Abstract base class for Physics-Informed Neural Network models (both deterministic and Bayesian).
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        pass

    def kl_divergence(self) -> torch.Tensor:
        """
        Return KL divergence term.
        For deterministic models, this can return 0.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 100) -> tuple:
        """
        Monte Carlo prediction to estimate mean and variance.
        By default, performs deterministic forward `n_samples` times.
        Override in Bayesian models where outputs change per sample.
        """
        preds = [self(x).detach() for _ in range(n_samples)]
        preds = torch.stack(preds)
        return preds.mean(dim=0), preds.var(dim=0)
