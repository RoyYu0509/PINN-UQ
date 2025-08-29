#Importing the necessary
import os
import numpy as np
import math
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

import torch
import torch.nn as nn
import sklearn
from sklearn.neighbors import NearestNeighbors
from utils_uqmd.utils_uq_vi import VIBPINN
from utils_uqmd.utils_uq_mlp import MLPPINN
from utils_uqmd.utils_uq_cp import CP

from scipy.stats import norm
from utils_uqmd.utils_uq_dropout import DropoutPINN


if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU (M1/M2/M3)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # Fallback to CPU


def generating_alphas(n: int = 20,
                      step: float = 0.05,
                      dtype=torch.float32) -> torch.Tensor:
    """
    Return `n` alpha levels evenly spaced by `step`, starting at `step`
    (so 0.0 is excluded) and ending at 1.0.

    By default (`n=20`, `step=0.05`) the tensor is:
        0.05, 0.10, …, 0.95, 1.00
    which ensures an exact 0.95 entry.

    Returns
    -------
    alphas : (n, 1) torch.Tensor
        Column vector of alpha values.
    """
    alphas = torch.arange(step, 1.0 + 1e-8, step, dtype=dtype)   # 0.05 … 1.00
    if len(alphas) != n:
        raise ValueError(f"With step={step}, you need n={len(alphas)} for range 0–1.")
    
    return alphas[:len(alphas)-1].view(-1, 1)

import torch

def _to_1d_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device).reshape(-1)
    # numpy or list
    return torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1)

def _normalize_predset(pred_set):
    lower, upper = pred_set
    device = lower.device if isinstance(lower, torch.Tensor) else (
        upper.device if isinstance(upper, torch.Tensor) else torch.device("cpu")
    )
    l = _to_1d_torch(lower, device)
    u = _to_1d_torch(upper, device)
    # ensure lower <= upper elementwise
    swap_mask = l > u
    if swap_mask.any():
        l_new = l.clone(); u_new = u.clone()
        l_new[swap_mask], u_new[swap_mask] = u[swap_mask], l[swap_mask]
        l, u = l_new, u_new
    return l, u

def _coverage(pred_set, y_true):
    """
    Empirical coverage: fraction of targets inside [lower, upper].
    Accepts tensors or numpy; shapes can be (N,), (N,1) — coerced to 1-D.
    """
    lower, upper = _normalize_predset(pred_set)
    y = _to_1d_torch(y_true, lower.device)
    # length guard
    assert y.numel() == lower.numel() == upper.numel(), \
        f"len mismatch: y={y.numel()}, lower={lower.numel()}, upper={upper.numel()}"
    inside = (y >= lower) & (y <= upper)
    return inside.float().mean().item()

def _sharpness(pred_set):
    """Average interval width (mean sharpness)."""
    lower, upper = _normalize_predset(pred_set)
    return (upper - lower).mean().item()

def _interval_score(pred_set, y_true, alpha):
    """
    Mean Interval Score (Gneiting & Raftery 2007).
    IS = (u-l) + (2/alpha)*(l - y)_+ + (2/alpha)*(y - u)_+
    """
    lower, upper = _normalize_predset(pred_set)
    y = _to_1d_torch(y_true, lower.device)
    assert y.numel() == lower.numel() == upper.numel(), \
        f"len mismatch: y={y.numel()}, lower={lower.numel()}, upper={upper.numel()}"
    width = (upper - lower)
    below_miss = (lower - y).clamp(min=0.0)
    above_miss = (y - upper).clamp(min=0.0)
    score = width + (2.0 / float(alpha)) * (below_miss + above_miss)
    return score.mean().item()  # mean, not sum (stable across subset sizes)


# Test coverage for VI model under different level of uncertainty
def vi_test_uncertainties(uqmodel, alphas, X_test, Y_test):
    """
        Evaluate uncertainty metrics (coverage and sharpness) over a range of alphas.

        Parameters:
            uqmodel: callable that returns (pred_set, empirical_coverage)
            alphas: list of uq uncertainty levels (CP:alpha; Drop-Out:drop_out_rate; VI:prior_std)
            X_test, Y_test: test data

        Returns:
            pandas.DataFrame with columns ["alpha", "coverage", "sharpness"]
    """
    if isinstance(uqmodel, VIBPINN):
        results = []

        for alpha in tqdm(alphas):
            alpha_val = float(alpha.item())
            if not (0.0 < alpha_val < 1.0):
                raise ValueError("alpha must be in (0,1) for VI.")
            pred_set = uqmodel.predict(alpha, X_test, n_samples=5000)
            coverage = _coverage(pred_set, Y_test)
            sha = _sharpness(pred_set)
            interval_score = _interval_score(pred_set, Y_test, alpha)

            results.append({
                "alpha": alpha_val,
                "coverage": coverage,
                "sharpness": sha,
                "interval score": interval_score
            })
        return pd.DataFrame(results)

    else:
        raise ValueError("The given model must be VI BPINN")


# Test CP model
def cp_test_uncertainties(uqmodel, alphas, X_test, Y_test, X_cal, Y_cal, X_train, Y_train, heuristic_u, k):
    """
    Test the given cp uq model, using different uq metrics
    """
    # if isinstance(uqmodel, CP):
    results=[]
    for alpha in tqdm(alphas):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1) for VI.")
        pred_set = uqmodel.predict(alpha, X_test,  X_train,  Y_train, X_cal, Y_cal, heuristic_u=heuristic_u, k=k)
        coverage = _coverage(pred_set, Y_test)
        sha = _sharpness(pred_set)
        interval_score = _interval_score(pred_set, Y_test, alpha)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sha,
            "interval score": interval_score
        })
    return pd.DataFrame(results)

    # else:
    #     raise ValueError("The given model must be CP PINN!")

# Test Drop Out model
def do_test_uncertainties(uqmodel, alphas, X_test, Y_test, n_samples):
    """
    Test the given drop-out uq model, using different uq metrics
    """
    if isinstance(uqmodel, DropoutPINN):
        results=[]
        for alpha in tqdm(alphas):
            alpha_val = float(alpha)
            if not (0.0 < alpha_val < 1.0):
                raise ValueError("alpha must be in (0,1) for VI.")
            pred_set = uqmodel.predict(alpha, X_test, n_samples,)
            coverage = _coverage(pred_set, Y_test)
            sha = _sharpness(pred_set)
            interval_score = _interval_score(pred_set, Y_test, alpha)

            results.append({
                "alpha": alpha_val,
                "coverage": coverage,
                "sharpness": sha,
                "interval score": interval_score
            })
        return pd.DataFrame(results)

    else:
        raise ValueError("The given model must be Dropout PINN!")


# ---------------------------------------------------------------------
#  Test coverage / sharpness for Hamiltonian-MC model
# ---------------------------------------------------------------------
from tqdm import tqdm
import pandas as pd

def hmc_test_uncertainties(uqmodel,
                           alphas,
                           X_test,
                           Y_test,
                           n_samples: int = 5000):
    """
    Evaluate uncertainty metrics (coverage and sharpness) for an HMC-based
    Bayesian PINN across a grid of α values.

    Parameters
    ----------
    uqmodel     : instance of HMCNN
    alphas      : iterable of α values in (0, 1)
    X_test,
    Y_test      : test set tensors
    n_samples   : how many posterior weight draws to use per α
                  (passed to uqmodel.predict)

    Returns
    -------
    pandas.DataFrame with columns ["alpha", "coverage", "sharpness"]
    """

    results = []
    for alpha in tqdm(alphas):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1) for HMC.")
        # Predict lower/upper bounds
        pred_set = uqmodel.predict(alpha_val, X_test, n_samples=n_samples)
        coverage = _coverage(pred_set, Y_test)
        sha      = _sharpness(pred_set)
        interval_score = _interval_score(pred_set, Y_test, alpha)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sha,
            "interval score": interval_score
        })
    return pd.DataFrame(results)


# Test Dist.based md
def dist_test_uncertainties(uqmodel,
                           alphas,
                           X_test,
                           Y_test,
                           heuristic_u: str = "features",
                           n_samples: int = 1000):
    """
    Evaluate uncertainty metrics (coverage and sharpness) for an HMC-based
    Bayesian PINN across a grid of α values.

    Parameters
    ----------
    uqmodel     : instance of HMCNN
    alphas      : iterable of α values in (0, 1)
    X_test,
    Y_test      : test set tensors
    n_samples   : how many posterior weight draws to use per α
                  (passed to uqmodel.predict)

    Returns
    -------
    pandas.DataFrame with columns ["alpha", "coverage", "sharpness"]
    """
    results = []
    for alpha in tqdm(alphas):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1) for HMC.")
        # Predict lower/upper bounds
        pred_set = uqmodel.predict(alpha_val, X_test, 
                                   heuristic_u=heuristic_u, n_samples=n_samples)
        coverage = _coverage(pred_set, Y_test)
        sha      = _sharpness(pred_set)
        interval_score = _interval_score(pred_set, Y_test, alpha)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sha,
            "interval score": interval_score
        })
    return pd.DataFrame(results)

