import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# Helper to convert torch Tensor to numpy (handles both Tensor and ndarray inputs)
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class AdaptiveCP:
    """
    Adaptive Conformal Predictor that learns a feature-dependent conformity threshold.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained regression model (must support .forward; if using latent heuristic, forward(return_hidden=True) should return (pred, hidden)).
    alpha : float, optional (default=0.05)
        Target miscoverage level (e.g., 0.05 for 95% coverage). A neural network is trained to predict the (1-alpha) quantile.
    device : torch.device or None, optional
        Device on which to run computations (defaults to model's device or CPU).
    heuristic : {'feature', 'latent', 'raw_std'}, optional (default='feature')
        The type of conformity score to use:
          - 'feature': distance in input feature space to k nearest neighbors.
          - 'latent': distance in model's latent space to k nearest neighbors.
          - 'raw_std': the model's own predictive interval width (requires model.predict method).
    k : int, optional (default=10)
        Number of neighbors for K-NN based heuristics ('feature' or 'latent').
    conf_nn_hidden_layers : tuple, optional (default=(64,64))
        Hidden layer sizes for the quantile prediction network.
    conf_nn_lr : float, optional (default=1e-3)
        Learning rate for training the quantile network.
    conf_nn_epochs : int, optional (default=100)
        Number of training conf_nn_epochs for the quantile network (pinball loss optimization).
    """
    def __init__(self, model, alpha=0.05, device=None, heuristic="feature",
                 conf_nn_hidden_layers=(64, 128, 64), conf_nn_lr=1e-4, conf_nn_epochs=10000):
        self.model = model
        self.alpha = alpha
        self.device = device or (next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu'))
        self.heuristic = heuristic
        self.conf_nn_hidden_layers = conf_nn_hidden_layers
        self.conf_nn_lr = conf_nn_lr
        self.conf_nn_epochs = conf_nn_epochs
        # Prepare model
        self.model.eval()  # ensure model is in eval mode
        # Quantile model will be initialized once we know input/output dimensions
        self.quantile_model = None
        self.quantile_model_trained = False

    # ---------- distance / width helpers ----------
    def _feature_distance(self, X_ref, X_train, k: int):
        nbrs = NearestNeighbors(n_neighbors=k).fit(_to_numpy(X_train))
        dist, _ = nbrs.kneighbors(_to_numpy(X_ref))
        return dist.mean(axis=1)

    def _latent_distance(self, X_ref, X_train, k):
        with torch.no_grad():
            H_ref = self.model(X_ref.to(self.device), return_hidden=True)[1]
            H_trn = self.model(X_train.to(self.device), return_hidden=True)[1]

        nbrs = NearestNeighbors(n_neighbors=k+1).fit(_to_numpy(H_trn))
        dist, _ = nbrs.kneighbors(_to_numpy(H_ref))
        # throw away the 0-distance self-neighbour
        return dist[:, 1:].mean(axis=1)

    def _rawstd_width(self, alpha, X):
        lower, upper = self.model.predict(alpha, X.to(self.device))
        return _to_numpy((upper - lower).squeeze(-1))

    # ---------- conformity scores (takes k) ----------
    def _compute_conformity_scores(self, X_cal, Y_cal, X_train, k: int):
        with torch.no_grad():
            Y_pred = self.model(X_cal.to(self.device))
        residual = np.abs(_to_numpy(Y_cal) - _to_numpy(Y_pred))

        if self.heuristic == "feature":
            scale = self._feature_distance(X_cal, X_train, k)
        elif self.heuristic == "latent":
            scale = self._latent_distance(X_cal, X_train, k)
        elif self.heuristic == "raw_std":
            scale = self._rawstd_width(self.alpha, X_cal)
        else:
            raise ValueError("Unknown heuristic.")
        scale = np.maximum(scale, 1e-8)
        return residual / scale[:, None]

    # ---------- quantile-network helpers (unchanged logic) ----------
    def _init_quantile_model(self, in_dim, out_dim):
        layers, prev = [], in_dim
        for h in self.conf_nn_hidden_layers:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.quantile_model = torch.nn.Sequential(*layers).to(self.device)

    def _train_quantile_model(self, X_feat, scores):
        self.quantile_model.train()
        opt = torch.optim.Adam(self.quantile_model.parameters(), lr=self.conf_nn_lr)
        τ = 1.0 - self.alpha

        def pinball(pred, true):
            diff = true - pred
            return torch.where(diff >= 0, τ * diff, (τ - 1) * diff).mean()

        X_feat, scores = X_feat.to(self.device), scores.to(self.device)
        for _ in range(self.conf_nn_epochs):
            opt.zero_grad()
            loss = pinball(self.quantile_model(X_feat), scores)
            loss.backward()
            opt.step()

        self.quantile_model.eval()
        self.quantile_model_trained = True

    # ---------- main API ----------
    def predict(
        self,
        alpha,
        X_test,
        *,
        X_train=None,
        Y_train=None,
        X_cal=None,
        Y_cal=None,
        heuristic_u=None,
        k: int = 10,              # <-- k is supplied per call (default 10)
    ):
        if alpha != self.alpha:
            print(f"[AdaptiveCP] Warning: using α={self.alpha}; requested {alpha}.")
        heuristic_u = heuristic_u or self.heuristic

        # -- train q-net the first time -----------------------------
        if not self.quantile_model_trained:
            if X_cal is None or Y_cal is None:
                raise ValueError("Pass X_cal/Y_cal on first call to train q-network.")
            if heuristic_u in ("feature", "raw_std"):
                X_feat = X_cal
            elif heuristic_u == "latent":
                with torch.no_grad():
                    X_feat = self.model(
                        X_cal.to(self.device), return_hidden=True
                    )[1].cpu()
            else:
                raise ValueError("Unknown heuristic.")
            scores = self._compute_conformity_scores(X_cal, Y_cal, X_train, k)
            X_feat_t = (
                X_feat
                if isinstance(X_feat, torch.Tensor)
                else torch.tensor(X_feat, dtype=torch.float32)
            )
            scores_t = torch.tensor(scores, dtype=torch.float32)
            self._init_quantile_model(X_feat_t.shape[1], scores_t.shape[1])
            print(f"\n[🟠] Training Adaptive CP Model")
            self._train_quantile_model(X_feat_t, scores_t)

        # -- q̂_α(x_test) -----------------------------
        X_test_t = (
            X_test.to(self.device)
            if isinstance(X_test, torch.Tensor)
            else torch.tensor(X_test, dtype=torch.float32, device=self.device)
        )
        if heuristic_u in ("feature", "raw_std"):
            X_feat_test = X_test_t
        elif heuristic_u == "latent":
            with torch.no_grad():
                X_feat_test = self.model(X_test_t, return_hidden=True)[1]
        else:
            raise ValueError("Unknown heuristic.")

        with torch.no_grad():
            q_hat = self.quantile_model(X_feat_test).cpu().numpy()

        # -- point prediction -------------------------
        with torch.no_grad():
            y_pred = self.model(X_test_t).cpu().numpy()

        # -- local scale ------------------------------
        if heuristic_u == "feature":
            if X_train is None:
                raise ValueError("X_train required for 'feature'.")
            scale = self._feature_distance(X_test_t, X_train, k)
        elif heuristic_u == "latent":
            if X_train is None:
                raise ValueError("X_train required for 'latent'.")
            scale = self._latent_distance(X_test_t, X_train, k)
        elif heuristic_u == "raw_std":
            scale = self._rawstd_width(self.alpha, X_test_t)
        else:
            raise ValueError("Unknown heuristic.")
        scale = np.maximum(scale, 1e-8)

        # -- assemble intervals -----------------------
        eps = q_hat * (scale[:, None] if q_hat.ndim > 1 else scale)
        lower = torch.tensor(y_pred - eps, dtype=torch.float32, device=self.device)
        upper = torch.tensor(y_pred + eps, dtype=torch.float32, device=self.device)
        return [lower, upper]






import pandas as pd
from tqdm import tqdm
from utils_tools.utils_result_metrics import _coverage, _sharpness, _sdcv, _interval_score

def adaptive_cp_test_uncertainties_grid(
    base_md,            # a *trained* torch.nn.Module
    alphas,                    # iterable of α values (0<α<1)
    X_test,  Y_test,           # held-out set
    X_cal,   Y_cal,            # calibration set
    X_train, Y_train=None,     # training set (for distance heuristics)
    *,
    heuristic_u="feature",     # 'feature' | 'latent' | 'raw_std'
    k=10,                      # k-NN size
    conf_nn_hidden_layers=(64,64),
    conf_nn_lr=1e-3,
    conf_nn_epochs=100,
):
    """
    For *each* α:
      • instantiates AdaptiveCP(alpha=α)
      • trains its quantile network on (X_cal, Y_cal)
      • builds prediction intervals on X_test
      • logs coverage / sharpness / sdcv / interval-score

    Returns
    -------
    pd.DataFrame  one row per α
    """
    results = []

    for alpha in tqdm(alphas, desc="Adaptive-CP grid"):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1).")

        # ── 1. new AdaptiveCP tailored to this α ───────────────────────────
        acp = AdaptiveCP(
            base_md,
            alpha=alpha_val,
            heuristic=heuristic_u,
            conf_nn_hidden_layers=conf_nn_hidden_layers,
            conf_nn_lr=conf_nn_lr,
            conf_nn_epochs=conf_nn_epochs,
        )

        # ── 2. train quantile-net & build intervals ───────────────────────
        pred_set = acp.predict(
            alpha_val, X_test,
            X_train=X_train, Y_train=Y_train,
            X_cal=X_cal,   Y_cal=Y_cal,
            heuristic_u=heuristic_u, k=k
        )

        # ── 3. diagnostics ────────────────────────────────────────────────
        cov   = _coverage(pred_set, Y_test)
        sharp = _sharpness(pred_set)
        sdcv  = _sdcv(pred_set)
        iscore= _interval_score(pred_set, Y_test, alpha_val)

        results.append({
            "alpha"         : alpha_val,
            "coverage"      : cov,
            "sharpness"     : sharp,
            "sdcv"          : sdcv,
            "interval score": iscore,
        })

    return pd.DataFrame(results)