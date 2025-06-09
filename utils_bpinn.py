from interface_model import BasePINNModel
from utils_BayesianLinearLayer import BayesianLinearLayer as BayesianLinear

import torch
import torch.nn as nn
import math

class BayesianFeedForwardPINN(BasePINNModel):
    """Feed-forward neural network with Bayesian linear layers (for VI)."""
    def __init__(self, input_dim, hidden_dims, output_dim, prior_std=1.0, act_func=nn.Tanh()):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        prev_dim = input_dim
        # Build hidden layers with BayesianLinear
        for h in hidden_dims:
            layers.append(BayesianLinear(prev_dim, h, prior_std=prior_std))  # in_feat, out_feat, prior_std
            layers.append(act_func)
            prev_dim = h
        # Final output layer (Bayesian linear as well)
        layers.append(BayesianLinear(in_features = prev_dim, out_features = output_dim, prior_std=prior_std))
        self.layers = nn.ModuleList(layers)  # not using Sequential because it's a mix of custom and activations

    def forward(self, x):
        out = x
        for layer in self.layers:
            # [BL, act, BL, act, ..., act, BL] go through all the layers
            out = layer(out)  # BayesianLinear or activation
        return out

    def kl_divergence(self):
        # Sum KL from all BayesianLinear layers
        kl_total = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_total += layer.kl_divergence()
        return kl_total
