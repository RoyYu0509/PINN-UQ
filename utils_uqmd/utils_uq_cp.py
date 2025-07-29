# ─────────────────────────────────────────────────────────────────────
#  🆕  Conformal Predictor (CP) – drop-in replacement
# ─────────────────────────────────────────────────────────────────────
import numpy as np
from numpy import random
import torch
from sklearn.neighbors import NearestNeighbors
from utils_uqmd.utils_uq_hmc import HMCBPINN

# 🔹 tiny helper: robust Torch/NumPy conversion
def _to_numpy(x):
    """Return a NumPy array regardless of whether *x* is Tensor or ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class CP:
    """
    Conformal predictor wrapper supporting three heuristics:
      • 'feature'  – k-NN distance in input space
      • 'latent'   – k-NN distance in hidden space (model must return_hidden=True)
      • 'raw_std'  – raw predictive interval width from the model itself
    """

    def __init__(self, model, device=None):
        self.model  = model
        self.device = device or next(model.parameters()).device
        # 1️⃣ put the model in eval mode once and for all
        self.model.eval()

    # ════════════════════════════════════════════════════════════════
    # 1️⃣  k-NN helper functions
    # ════════════════════════════════════════════════════════════════
    def _feature_distance(self, X_test, X_train, k):
        X_test_np = X_test.clone().detach()
        mean = self.model(X_test_np.to(self.device))
        X_test = _to_numpy(X_test)
        X_train = _to_numpy(X_train)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
        distances, _ = nbrs.kneighbors(X_test)
        return distances.mean(axis=1), mean              # (N_cal,)

    def _latent_distance(self, X_test, X_train, k):
        X_test_np = X_test.clone().detach()
        with torch.no_grad():
            mean = self.model(X_test_np.to(self.device))
        self.model.eval()
        with torch.no_grad():
            H_cal = self.model(X_test.to(self.device),   return_hidden=True)[1]
            H_trn = self.model(X_train.to(self.device), return_hidden=True)[1]
        H_cal   = _to_numpy(H_cal)
        H_trn   = _to_numpy(H_trn)
        nbrs = NearestNeighbors(n_neighbors=k).fit(H_trn)
        distances, _ = nbrs.kneighbors(H_cal)

        return distances.mean(axis=1), mean

    # ════════════════════════════════════════════════════════════════
    # 2️⃣  Raw predictive width
    # ════════════════════════════════════════════════════════════════
    def _rawstd(self, alpha, X):
        """Return (upper-lower) from the model’s own `predict` method."""
        self.model.eval()
        with torch.no_grad():
            pred_set = self.model.predict(alpha, X)
        y_pred = (pred_set[1] + pred_set[0])/2
        width = (pred_set[1] - pred_set[0]).squeeze(-1)    # (N,)
        return _to_numpy(width), y_pred.cpu().numpy()

    # ════════════════════════════════════════════════════════════════
    # 3️⃣  Conformal scores on the *calibration* set
    # ════════════════════════════════════════════════════════════════
    def _conf_metric_feature(self, X_cal, Y_cal, X_train, k, eps=1e-8):
        with torch.no_grad():
            Y_pred = self.model(X_cal.to(self.device)).cpu().numpy()
        residual   = np.abs(Y_cal - Y_pred)                # (N_cal, out_dim)
        dist_feat_tensor, _ = self._feature_distance(X_cal, X_train, k)
        dist_feat = np.maximum(_to_numpy(dist_feat_tensor), eps)
        return residual / dist_feat[:, None]               # (N_cal, out_dim)

    def _conf_metric_latent(self, X_cal, Y_cal, X_train, k, eps=1e-8):
        with torch.no_grad():
            Y_pred = self.model(X_cal.to(self.device)).cpu().numpy()
        residual   = np.abs(Y_cal - Y_pred)
        dist_lat_tensor, _ = self._latent_distance(X_cal, X_train, k)
        dist_lat = np.maximum(_to_numpy(dist_lat_tensor), eps)
        return residual / dist_lat[:, None]

    def _conf_metric_rawstd(self, alpha, X_cal, Y_cal, X_train, eps=1e-8):
        with torch.no_grad():
            pred_set = self.model.predict(alpha, X_cal.to(self.device))
        hi = pred_set[1].cpu().numpy()
        lo = pred_set[0].cpu().numpy()
        Y_pred = (hi + lo)/2
        width = np.maximum((hi - lo).squeeze(-1), eps)
        residual   = np.abs(Y_cal - Y_pred)
        # width      = np.maximum(self._rawstd(alpha, X_cal), eps)    # (N_cal,)s
        assert residual.shape[0] == width.shape[0], "Residual/width length mismatch"
        return residual / width[:, None]

    # ════════════════════════════════════════════════════════════════
    # 4️⃣  Public API
    # ════════════════════════════════════════════════════════════════
    def predict(
        # ------------ args ---------------
        self, alpha,
        X_test,  
        # ----------- kwargs ---------------
        X_train=None,  Y_train=None,
        X_cal=None,   Y_cal=None,
        heuristic_u="feature",
        k=10
    ):
        """
        Parameters
        ----------
        alpha : float      desired mis-coverage (e.g. 0.05)
        heuristic_u : str  'feature' | 'latent' | 'raw_std'
        k : int            nearest-neighbour count for k-NN heuristics
        """
        # 0️⃣  deterministic run
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        torch.use_deterministic_algorithms(True)
        self.model.eval()

        # --- choose conformity metric ----------------------------------------
        if heuristic_u == "feature":
            cal_scores = self._conf_metric_feature(X_cal, Y_cal, X_train, k)
            test_u, mean    = self._feature_distance(X_test, X_train, k)
        elif heuristic_u == "latent":
            cal_scores = self._conf_metric_latent(X_cal, Y_cal, X_train, k)
            test_u, mean   = self._latent_distance(X_test, X_train, k)
        elif heuristic_u == "raw_std":
            cal_scores = self._conf_metric_rawstd(alpha, X_cal, Y_cal, X_train)
            test_u, mean     = self._rawstd(alpha, X_test)
        else:
            raise ValueError("heuristic_u must be 'feature', 'latent' or 'raw_std'")

        # --- quantile on calibration scores ----------------------------------
        n_cal = cal_scores.shape[0]

        # Compute the conformal quantile level
        q = np.ceil((n_cal + 1) * (1 - float(alpha))) / n_cal
        q = np.clip(q, 0.0, 1.0 - 1e-12)     # keep within [0,1)

        q_hat = np.quantile(
            cal_scores,
            q,
            axis=0,
            method="higher"
        )
                                       # (out_dim,)

        # # --- point predictions on X_test --------------------------------------
        # with torch.no_grad():
        #     self.model.eval()
        #     y_pred_test = self.model(X_test.to(self.device)).cpu().numpy()
            
        # --- build intervals --------------------------------------------------
        eps  = q_hat * test_u[:, None]                   # (N_test, out_dim)
        with torch.no_grad():
            if isinstance(mean, torch.Tensor):
                lower = (mean - torch.tensor(eps, dtype=mean.dtype, device=self.device))
                upper = (mean + torch.tensor(eps, dtype=mean.dtype, device=self.device))
            else:
                lower = torch.tensor(mean - eps, dtype=torch.float32, device=self.device)
                upper = torch.tensor(mean + eps, dtype=torch.float32, device=self.device)

        del cal_scores, test_u, mean, q_hat, eps
        import gc
        gc.collect()
        torch.mps.empty_cache()

        return (lower, upper)