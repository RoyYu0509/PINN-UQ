from utils_uqmd.interface_layer import BaseLayer
import torch
import torch.nn as nn
import math



# def default_mu_rho(in_features, out_features,
#                    mu_std=0.1, rho=-3.0,
#                    mu_mean=0.0,
#                    prior_std=1.0):

#     # Weights and Biases Distribution Initialization
#     weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(mu_mean, mu_std))
#     weight_rho = nn.Parameter(torch.empty(out_features, in_features).fill_(rho))
#     bias_mu = nn.Parameter(torch.empty(out_features).normal_(mu_mean, mu_std))
#     bias_rho = nn.Parameter(torch.empty(out_features).fill_(rho))

#     # Std of the prior distribution
#     prior_std = prior_std

#     return weight_mu, weight_rho, bias_mu, bias_rho, prior_std


import torch
import torch.nn as nn

def default_mu_rho(in_features, out_features,
                  mu_std=0.1,        # accepted for API compatibility, unused
                  rho=-3.0,
                  prior_std=1.0,
                  gain=None):
    if gain is None:
        # if using tanh activations downstream, this is recommended
        gain = nn.init.calculate_gain('tanh')  # 5/3

    # Weight mean
    weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    nn.init.xavier_uniform_(weight_mu, gain=gain)

    # Bias mean: typically start from zero
    bias_mu = nn.Parameter(torch.zeros(out_features))

    # Rho parameters (pre-softplus)
    weight_rho = nn.Parameter(torch.full((out_features, in_features), rho))
    bias_rho   = nn.Parameter(torch.full((out_features,), rho))

    return weight_mu, weight_rho, bias_mu, bias_rho, float(prior_std)


class BayesianLinearLayer(BaseLayer):
    """
    Fully-factorised Bayesian linear layer.
    Call with  sample=False  to obtain the posterior *mean* (deterministic).
    """

    def __init__(self, in_features, out_features,
                 mu_std=0.1, rho=-3.0, prior_std=1.0,
                 initialization=default_mu_rho):
        super().__init__()
        (self.weight_mu, self.weight_rho,
         self.bias_mu,  self.bias_rho,
         self.prior_std) = initialization(in_features, out_features,
                                          mu_std=mu_std, rho=rho,
                                          prior_std=prior_std)
        self.log2pi = math.log(2 * math.pi)

    # ---------- helpers ----------
    @staticmethod
    def _softplus(x):
        return torch.log1p(torch.exp(x))

    # ---------- forward ----------
    def forward(self, x, *, sample: bool = True):
        """
        When sample=True (default) draw a new weight/bias sample.
        When sample=False use only the posterior means for deterministic output.
        """
        if sample:
            w_sigma = self._softplus(self.weight_rho)
            b_sigma = self._softplus(self.bias_rho)
            eps_w   = torch.randn_like(w_sigma)
            eps_b   = torch.randn_like(b_sigma)
            weight  = self.weight_mu + w_sigma * eps_w
            bias    = self.bias_mu  + b_sigma * eps_b
        else:                       # posterior mean
            weight, bias = self.weight_mu, self.bias_mu
        return x.matmul(weight.t()) + bias

    # ---------- KL divergence ----------
    def kl_divergence(self):
        w_sigma = self._softplus(self.weight_rho)
        b_sigma = self._softplus(self.bias_rho)
        prior_var = self.prior_std ** 2

        kl_w = torch.sum(torch.log(self.prior_std / w_sigma) +
                         0.5 * (w_sigma ** 2 + self.weight_mu ** 2) / prior_var - 0.5)
        kl_b = torch.sum(torch.log(self.prior_std / b_sigma) +
                         0.5 * (b_sigma ** 2 + self.bias_mu ** 2) / prior_var - 0.5)
        return kl_w + kl_b


# Explain::
# class BayesianLinearLayer(BaseLayer):
#     """ 一层 Bayesian Linear Layer
#         Bayesian Linear layer with Gaussian weight and bias priors and variational posteriors.
#     """

#     def __init__(self, in_features, out_features, mu_std, rho, prior_std, initialization=default_mu_rho):
#         super().__init__()
#         # Mean and log-variance (or rho) for weights and biases as learnable parameters
#         self.in_features = in_features
#         self.out_features = out_features

#         # ------------------------------ Model's Parameters ------------------------------------------
#         # Initialize means (mu) to small random values, and rho to a small negative (so sigma ~ small)
#         # Since σ must be strictly positive, so we optimize rho, compute σ by softplus(rho)
#         # So, we are still learning the std σ, but indirectly
#         (self.weight_mu, self.weight_rho, self.bias_mu, self.bias_rho,
#          self.prior_std) = initialization(in_features, out_features, mu_std=mu_std, rho=rho, prior_std=prior_std)

#         # Prior standard deviation (fixed)
#         self.log2pi = math.log(2 * math.pi)  # for potential use in exact logprob if needed

#     def forward(self, x):
#         # Sample the std  σ  of the weights and biases (the reparameterization trick)
#         weoight_sigma = torh.log1p(torch.exp(self.weight_rho))  # softplus to ensure positivity
#         bias_sigma = torch.log1p(torch.exp(self.bias_rho))

#         # Sample ε ∼ 𝒩(0,1) for weights and baises
#         eps_w = torch.randn_like(weight_sigma)
#         eps_b = torch.randn_like(bias_sigma)

#         # Sample from 𝒩(mu, sigma^2) through variable transformation
#         # 这样, 我们就能 update `mu` 和 `sigma`(rho)
#         weight = self.weight_mu + weight_sigma * eps_w
#         bias = self.bias_mu + bias_sigma * eps_b

#         # Linear layer computation xWᵀ + b
#         return x.matmul(weight.t()) + bias  # the utput of this bayesian linear layer

#     def kl_divergence(self):
#         # Compute KL divergence KL[q(w,b) || p(w,b)] for this layer (sum over all weights and biases)
#         # Assuming factorized Gaussian posteriors and Gaussian priors N(0, prior_std^2):contentReference[oaicite:2]{index=2}.
#         weight_sigma = torch.log1p(torch.exp(self.weight_rho))
#         bias_sigma = torch.log1p(torch.exp(self.bias_rho))
#         # KL for each weight: log(prior_sigma/post_sigma) + (post_sigma^2 + mu^2)/(2*prior_sigma^2) - 0.5
#         prior_var = self.prior_std ** 2

#         # Compute the KL value using the formula for Gaussian
#         #   = log(prior_std/posterior_std) + 0.5 * (posterior_std**2 + posterior_mu**2) / [prior_std^2] - 1)
#         # For numerical stability, avoid log of 0 by using weight_sigma (softplus ensures >0)
#         kl_weight = torch.sum(torch.log(self.prior_std / weight_sigma) +
#                               0.5 * (weight_sigma ** 2 + self.weight_mu ** 2) / prior_var - 0.5)
#         kl_bias = torch.sum(torch.log(self.prior_std / bias_sigma) +
#                             0.5 * (bias_sigma ** 2 + self.bias_mu ** 2) / prior_var - 0.5)
#         return kl_weight + kl_bias