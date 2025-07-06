from utils_uqmd.interface_layer import BaseLayer
import torch
import torch.nn as nn
import math

def kaiming_init(weights, bias, a):
    # Initialize weights and bias (similar to PyTorch default)
    nn.init.kaiming_uniform_(weights, a)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)

class DeterministicLinear(nn.Module):
    """A standard linear layer with deterministic weights and biases, like nn.Linear."""
    def __init__(self, in_features, out_features, initialization=kaiming_init):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Create learnable parameters: weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Weights initialization
        initialization(self.weight, self.bias, math.sqrt(5))


    def forward(self, x):
        return x.matmul(self.weight.t()) + self.bias

    def kl_divergence(self):
        """
        Return zero KL divergence for deterministic layers.
        This ensures compatibility with shared Bayesian/DNN training pipelines.
        """
        return torch.tensor(0.0, device=self.weight.device)
