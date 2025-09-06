import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from utils_tools.utils_result_metrics import _coverage, _sharpness, _interval_score

# Helper to convert torch Tensor to numpy (handles both Tensor and ndarray inputs)
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class AdaptiveCP:
    """
    """
    def __init__(self, model, alpha=0.05, device=None, heuristic="feature",
                 conf_nn_hidden_layers=(64, 64, 64), conf_nn_lr=5e-4, conf_nn_epochs=20000,
                 training_kwargs={"step_size":5000, "gamma":0.5}, quant_seed=None):
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
        self.step_size = training_kwargs["step_size"]
        self.gamma = training_kwargs["gamma"]


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

    def _train_quantile_model(self, X_feat, scores, step_size=None, gamma=0.9):
        if step_size is None:
            step_size = self.conf_nn_epochs // 5
        self.quantile_model.train()
        opt = torch.optim.Adam(self.quantile_model.parameters(), lr=self.conf_nn_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

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
            scheduler.step()  # decay LR every `step_size` epochs


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
        X_cal=None,            # OPTIONAL: used here for conformal calibration only
        Y_cal=None,            # OPTIONAL: used here for conformal calibration only
        heuristic_u=None,
        k: int = 10,
    ):
        if alpha != self.alpha:
            print(f"[AdaptiveCP] Warning: using α={self.alpha}; requested {alpha}.")
            raise ValueError("the given alpha must be the same as the model's definition")

        heuristic_u = heuristic_u or self.heuristic

        # ---------- train q-net ON TRAIN SET (once) ----------
        if not self.quantile_model_trained:
            if X_train is None or Y_train is None:
                raise ValueError("Pass X_train/Y_train to train the quantile regressor on first call.")

            # features for q-net on TRAIN
            X_train_t = X_train.to(self.device) if isinstance(X_train, torch.Tensor) \
                        else torch.tensor(X_train, dtype=torch.float32, device=self.device)

            if heuristic_u == "feature":
                X_feat_tr = X_train_t
            elif heuristic_u == "latent":
                with torch.no_grad():
                    X_feat_tr = self.model(X_train_t, return_hidden=True)[1].cpu()
            elif heuristic_u == "raw_std":
                width_tr = torch.tensor(self._rawstd_width(self.alpha, X_train_t),
                                        dtype=torch.float32, device=self.device)
                X_feat_tr = torch.cat([X_train_t, width_tr.unsqueeze(-1)], dim=1).cpu()
            else:
                raise ValueError("Unknown heuristic.")

            # residuals on TRAIN
            with torch.no_grad():
                Y_pred_tr = self.model(X_train_t).cpu().numpy()
            residual_tr = np.abs(_to_numpy(Y_train) - Y_pred_tr)

            # scale on TRAIN (drop self-neighbour for KNN)
            if heuristic_u == "feature":
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(_to_numpy(X_train))
                dist_tr, _ = nbrs.kneighbors(_to_numpy(X_train))
                scale_tr = dist_tr[:, 1:].mean(axis=1)
            elif heuristic_u == "latent":
                scale_tr = self._latent_distance(X_train_t, X_train, k)
            elif heuristic_u == "raw_std":
                scale_tr = self._rawstd_width(self.alpha, X_train_t)
            else:
                raise ValueError("Unknown heuristic.")
            scale_tr = np.maximum(scale_tr, 1e-8)

            # pinball targets on TRAIN
            scores_tr = residual_tr / (scale_tr[:, None] if residual_tr.ndim > 1 else scale_tr)

            # train q-net
            X_feat_t = X_feat_tr if isinstance(X_feat_tr, torch.Tensor) \
                    else torch.tensor(X_feat_tr, dtype=torch.float32)
            scores_t = torch.tensor(scores_tr, dtype=torch.float32)
            self._init_quantile_model(X_feat_t.shape[1], scores_t.shape[1] if scores_t.ndim > 1 else 1)
            print(f"\n[🟠] Training Adaptive CP Quantile Net on TRAIN set: {scores_t.shape[0]} samples")
            self._train_quantile_model(X_feat_t.to(self.device), scores_t.to(self.device), 
                                       step_size=self.step_size,
                                       gamma=self.gamma)

        # ---------- q̂_α(x_test) and base prediction ----------
        X_test_t = X_test.to(self.device) if isinstance(X_test, torch.Tensor) \
                else torch.tensor(X_test, dtype=torch.float32, device=self.device)
        if heuristic_u == "feature":
            X_feat_test = X_test_t
        elif heuristic_u == "raw_std":
            width_test = torch.tensor(self._rawstd_width(self.alpha, X_test_t), dtype=torch.float32)
            X_feat_test = torch.cat([X_test_t, width_test.unsqueeze(-1).to(X_test_t.device)], dim=1)
        elif heuristic_u == "latent":
            with torch.no_grad():
                X_feat_test = self.model(X_test_t, return_hidden=True)[1]
        else:
            raise ValueError("Unknown heuristic.")

        with torch.no_grad():
            q_hat = self.quantile_model(X_feat_test).cpu().numpy()
            y_pred = self.model(X_test_t).cpu().numpy()
        q_hat = np.maximum(q_hat, 0.0)

        # ---------- local scale at TEST (relative to TRAIN set) ----------
        if heuristic_u == "feature":
            if X_train is None: raise ValueError("X_train required for 'feature'.")
            scale = self._feature_distance(X_test_t, X_train, k)
        elif heuristic_u == "latent":
            if X_train is None: raise ValueError("X_train required for 'latent'.")
            scale = self._latent_distance(X_test_t, X_train, k)
        elif heuristic_u == "raw_std":
            scale = self._rawstd_width(self.alpha, X_test_t)
        else:
            raise ValueError("Unknown heuristic.")
        scale = np.maximum(scale, 1e-8)

        # ---------- NEW: conformal calibration multiplier ĉ ----------
        c = 1.0
        if (X_cal is not None) and (Y_cal is not None):
            X_cal_t = X_cal.to(self.device) if isinstance(X_cal, torch.Tensor) \
                    else torch.tensor(X_cal, dtype=torch.float32, device=self.device)

            # q̂ on calibration features (built same way as test)
            if heuristic_u == "feature":
                X_feat_cal = X_cal_t
            elif heuristic_u == "raw_std":
                width_cal = torch.tensor(self._rawstd_width(self.alpha, X_cal_t), dtype=torch.float32)
                X_feat_cal = torch.cat([X_cal_t, width_cal.unsqueeze(-1)], dim=1)
            elif heuristic_u == "latent":
                with torch.no_grad():
                    X_feat_cal = self.model(X_cal_t, return_hidden=True)[1]
            else:
                raise ValueError("Unknown heuristic.")

            with torch.no_grad():
                q_cal = self.quantile_model(X_feat_cal).cpu().numpy()
                y_cal_pred = self.model(X_cal_t).cpu().numpy()

            # scale for calibration points (use TRAIN as neighbor pool)
            if heuristic_u == "feature":
                scale_cal = self._feature_distance(X_cal_t, X_train, k)
            elif heuristic_u == "latent":
                scale_cal = self._latent_distance(X_cal_t, X_train, k)
            elif heuristic_u == "raw_std":
                scale_cal = self._rawstd_width(self.alpha, X_cal_t)
            scale_cal = np.maximum(scale_cal, 1e-8)

            # normalized residuals and split-CP quantile
            R = np.abs(_to_numpy(Y_cal) - y_cal_pred) / (
                q_cal * (scale_cal[:, None] if q_cal.ndim > 1 else scale_cal)
            )
            R_flat = R if R.ndim == 1 else R.max(axis=1)  # conservative for multi-output
            m = R_flat.shape[0]
            k_idx = int(np.ceil((m + 1) * (1 - self.alpha))) - 1
            c = float(np.partition(R_flat, k_idx)[k_idx])

        # ---------- assemble intervals (conformalized if ĉ computed) ----------
        eps = c * q_hat * (scale[:, None] if q_hat.ndim > 1 else scale)
        lower = torch.tensor(y_pred - eps, dtype=torch.float32, device=self.device)
        upper = torch.tensor(y_pred + eps, dtype=torch.float32, device=self.device)
        return [lower, upper]


class AdaptiveCP_f:
    """
    Adaptive Conformal Prediction with a learned local quantile network (q-net).

    Drop-in replacement (no `deterministic_calls`):
    - Deterministic q-net init via `quant_seed` (default=12345).
    - Trains q-net once on first `predict()` call; reused afterwards.
    - Robust k-NN guards (k <= n_train-1 where needed).
    """
    def __init__(self, model, alpha=0.05, device=None, heuristic="feature",
                 conf_nn_hidden_layers=(64, 64, 64), conf_nn_lr=5e-4, conf_nn_epochs=20000,
                 training_kwargs={"step_size":5000, "gamma":0.5},
                 quant_seed: int | None = 12345):
        self.model = model
        self.alpha = alpha
        self.device = device or (next(model.parameters()).device
                                 if hasattr(model, 'parameters') else torch.device('cpu'))
        self.heuristic = heuristic
        self.conf_nn_hidden_layers = conf_nn_hidden_layers
        self.conf_nn_lr = conf_nn_lr
        self.conf_nn_epochs = conf_nn_epochs
        self.model.eval()  # inference mode for the base predictor
        self.quantile_model = None
        self.quantile_model_trained = False
        self.step_size = training_kwargs.get("step_size", max(1, conf_nn_epochs // 5))
        self.gamma = training_kwargs.get("gamma", 0.5)
        self.quant_seed = quant_seed

    # ---------- utilities ----------
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # ---------- distance / width helpers ----------
    def _feature_distance(self, X_ref, X_train, k: int):
        Xtr = self._to_numpy(X_train)
        n_tr = Xtr.shape[0]
        kk = max(1, min(int(k), n_tr))  # guard k
        nbrs = NearestNeighbors(n_neighbors=kk).fit(Xtr)
        dist, _ = nbrs.kneighbors(self._to_numpy(X_ref))
        return dist.mean(axis=1)

    def _latent_distance(self, X_ref, X_train, k):
        with torch.no_grad():
            H_ref = self.model(X_ref.to(self.device), return_hidden=True)[1]
            H_trn = self.model(X_train.to(self.device), return_hidden=True)[1]
        H_trn_np = self._to_numpy(H_trn)
        n_tr = H_trn_np.shape[0]
        kk = max(1, min(int(k) + 1, n_tr))  # +1 to drop the (approx) self neighbor
        nbrs = NearestNeighbors(n_neighbors=kk).fit(H_trn_np)
        dist, _ = nbrs.kneighbors(self._to_numpy(H_ref))
        if kk >= 2:
            return dist[:, 1:].mean(axis=1)
        else:
            return dist.mean(axis=1)

    def _rawstd_width(self, alpha, X):
        lower, upper = self.model.predict(alpha, X.to(self.device))
        return self._to_numpy((upper - lower).squeeze(-1))

    # ---------- conformity scores (kept for API completeness) ----------
    def _compute_conformity_scores(self, X_cal, Y_cal, X_train, k: int):
        with torch.no_grad():
            Y_pred = self.model(X_cal.to(self.device))
        residual = np.abs(self._to_numpy(Y_cal) - self._to_numpy(Y_pred))

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

    # ---------- quantile-network helpers ----------
    def _init_quantile_model(self, in_dim, out_dim):
        # Seed right before weight init to make re-instantiations identical.
        if self.quant_seed is not None:
            torch.manual_seed(self.quant_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.quant_seed)

        layers, prev = [], in_dim
        for h in self.conf_nn_hidden_layers:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.quantile_model = torch.nn.Sequential(*layers).to(self.device)

    def _train_quantile_model(self, X_feat, scores, step_size=None, gamma=0.9):
        if step_size is None:
            step_size = max(1, self.conf_nn_epochs // 5)
        self.quantile_model.train()
        opt = torch.optim.Adam(self.quantile_model.parameters(), lr=self.conf_nn_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

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
            scheduler.step()

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
        X_cal=None,            # OPTIONAL: used here for conformal calibration only
        Y_cal=None,            # OPTIONAL: used here for conformal calibration only
        heuristic_u=None,
        k: int = 10,
    ):
        if alpha != self.alpha:
            raise ValueError("the given alpha must be the same as the model's definition")

        heuristic_u = heuristic_u or self.heuristic

        # ---------- train q-net ON TRAIN SET (once) ----------
        if not self.quantile_model_trained:
            if X_train is None or Y_train is None:
                raise ValueError("Pass X_train/Y_train to train the quantile regressor on first call.")

            # features for q-net on TRAIN
            X_train_t = X_train.to(self.device) if isinstance(X_train, torch.Tensor) \
                        else torch.tensor(X_train, dtype=torch.float32, device=self.device)

            if heuristic_u == "feature":
                X_feat_tr = X_train_t
            elif heuristic_u == "latent":
                with torch.no_grad():
                    X_feat_tr = self.model(X_train_t, return_hidden=True)[1].cpu()
            elif heuristic_u == "raw_std":
                width_tr = torch.tensor(self._rawstd_width(self.alpha, X_train_t),
                                        dtype=torch.float32, device=self.device)
                X_feat_tr = torch.cat([X_train_t, width_tr.unsqueeze(-1)], dim=1).cpu()
            else:
                raise ValueError("Unknown heuristic.")

            # residuals on TRAIN
            with torch.no_grad():
                Y_pred_tr = self.model(X_train_t).cpu().numpy()
            residual_tr = np.abs(self._to_numpy(Y_train) - Y_pred_tr)

            # guard k for TRAIN computations using KNN
            n_tr = self._to_numpy(X_train).shape[0]
            k_guard = max(1, min(int(k), n_tr - 1)) if n_tr > 1 else 1

            # scale on TRAIN
            if heuristic_u == "feature":
                nbrs = NearestNeighbors(n_neighbors=k_guard + 1).fit(self._to_numpy(X_train))
                dist_tr, _ = nbrs.kneighbors(self._to_numpy(X_train))
                scale_tr = dist_tr[:, 1:].mean(axis=1) if k_guard >= 1 else dist_tr.mean(axis=1)
            elif heuristic_u == "latent":
                scale_tr = self._latent_distance(X_train_t, X_train, k_guard)
            elif heuristic_u == "raw_std":
                scale_tr = self._rawstd_width(self.alpha, X_train_t)
            else:
                raise ValueError("Unknown heuristic.")
            scale_tr = np.maximum(scale_tr, 1e-8)

            # pinball targets on TRAIN
            scores_tr = residual_tr / (scale_tr[:, None] if residual_tr.ndim > 1 else scale_tr)

            # train q-net
            X_feat_t = X_feat_tr if isinstance(X_feat_tr, torch.Tensor) \
                       else torch.tensor(X_feat_tr, dtype=torch.float32)
            scores_t = torch.tensor(scores_tr, dtype=torch.float32)
            self._init_quantile_model(X_feat_t.shape[1], scores_t.shape[1] if scores_t.ndim > 1 else 1)
            print(f"\n[🟠] Training Adaptive CP Quantile Net on TRAIN set: {scores_t.shape[0]} samples")
            self._train_quantile_model(X_feat_t.to(self.device), scores_t.to(self.device),
                                       step_size=self.step_size, gamma=self.gamma)

        # ---------- q̂_α(x_test) and base prediction ----------
        X_test_t = X_test.to(self.device) if isinstance(X_test, torch.Tensor) \
                   else torch.tensor(X_test, dtype=torch.float32, device=self.device)

        if heuristic_u == "feature":
            X_feat_test = X_test_t
        elif heuristic_u == "raw_std":
            width_test = torch.tensor(self._rawstd_width(self.alpha, X_test_t), dtype=torch.float32)
            X_feat_test = torch.cat([X_test_t, width_test.unsqueeze(-1).to(X_test_t.device)], dim=1)
        elif heuristic_u == "latent":
            with torch.no_grad():
                X_feat_test = self.model(X_test_t, return_hidden=True)[1]
        else:
            raise ValueError("Unknown heuristic.")

        with torch.no_grad():
            q_hat = self.quantile_model(X_feat_test).cpu().numpy()
            y_pred = self.model(X_test_t).cpu().numpy()
        q_hat = np.maximum(q_hat, 0.0)

        # ---------- local scale at TEST (relative to TRAIN set) ----------
        if heuristic_u in ("feature", "latent"):
            if X_train is None:
                raise ValueError("X_train required for 'feature' or 'latent' heuristic.")
            n_tr = self._to_numpy(X_train).shape[0]
            k_guard = max(1, min(int(k), n_tr - 1)) if n_tr > 1 else 1
        else:
            k_guard = 1

        if heuristic_u == "feature":
            scale = self._feature_distance(X_test_t, X_train, k_guard)
        elif heuristic_u == "latent":
            scale = self._latent_distance(X_test_t, X_train, k_guard)
        elif heuristic_u == "raw_std":
            scale = self._rawstd_width(self.alpha, X_test_t)
        else:
            raise ValueError("Unknown heuristic.")
        scale = np.maximum(scale, 1e-8)

        # ---------- conformal calibration multiplier ĉ ----------
        c = 1.0
        if (X_cal is not None) and (Y_cal is not None):
            X_cal_t = X_cal.to(self.device) if isinstance(X_cal, torch.Tensor) \
                      else torch.tensor(X_cal, dtype=torch.float32, device=self.device)

            # q̂ on calibration features (built same way as test)
            if heuristic_u == "feature":
                X_feat_cal = X_cal_t
            elif heuristic_u == "raw_std":
                width_cal = torch.tensor(self._rawstd_width(self.alpha, X_cal_t), dtype=torch.float32)
                X_feat_cal = torch.cat([X_cal_t, width_cal.unsqueeze(-1)], dim=1)
            elif heuristic_u == "latent":
                with torch.no_grad():
                    X_feat_cal = self.model(X_cal_t, return_hidden=True)[1]
            else:
                raise ValueError("Unknown heuristic.")

            with torch.no_grad():
                q_cal = self.quantile_model(X_feat_cal).cpu().numpy()
                y_cal_pred = self.model(X_cal_t).cpu().numpy()

            # scale for calibration points (use TRAIN as neighbor pool)
            if heuristic_u == "feature":
                scale_cal = self._feature_distance(X_cal_t, X_train, k_guard)
            elif heuristic_u == "latent":
                scale_cal = self._latent_distance(X_cal_t, X_train, k_guard)
            elif heuristic_u == "raw_std":
                scale_cal = self._rawstd_width(self.alpha, X_cal_t)
            else:
                raise ValueError("Unknown heuristic.")
            scale_cal = np.maximum(scale_cal, 1e-8)

            # normalized residuals and split-CP quantile
            R = np.abs(self._to_numpy(Y_cal) - y_cal_pred) / (
                q_cal * (scale_cal[:, None] if q_cal.ndim > 1 else scale_cal)
            )
            R_flat = R if R.ndim == 1 else R.max(axis=1)  # conservative for multi-output
            m = R_flat.shape[0]
            k_idx = max(0, min(m - 1, int(np.ceil((m + 1) * (1 - self.alpha))) - 1))  # guard
            c = float(np.partition(R_flat, k_idx)[k_idx]) if m > 0 else 1.0

        # ---------- assemble intervals ----------
        eps = c * q_hat * (scale[:, None] if q_hat.ndim > 1 else scale)
        lower = torch.tensor(y_pred - eps, dtype=torch.float32, device=self.device)
        upper = torch.tensor(y_pred + eps, dtype=torch.float32, device=self.device)
        return [lower, upper]




import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def cp_test_uncertainties_in_noisy_region(
    uqmodel, alphas,
    X_train, Y_train,
    X_cal, Y_cal,
    X_test, Y_test,
    idx_noisy_test,
    heuristic_u=None, k=None,
    *,                      # NEW: keyword-only for new args below
    compt_smooth=False      # NEW: evaluate on complement of idx_noisy_test when True
):
    """
    Train uqmodel on all available data (train + cal), then compute CP metrics on a subset
    of X_test determined by `idx_noisy_test` (default) or its complement (when compt_smooth=True).

    Parameters
    ----------
    uqmodel : CP model with .fit(...) and .predict(...)
    alphas : list of float (0 < alpha < 1)
    X_train, Y_train, X_cal, Y_cal, X_test, Y_test : torch.Tensor
    idx_noisy_test : indices or boolean mask specifying a subset of X_test.
        Accepts: slice | list/tuple | np.ndarray (bool or int) | torch.Tensor (bool or long)
    heuristic_u, k : forwarded to uqmodel.predict
    compt_smooth : bool
        False (default): evaluate on idx_noisy_test
        True:            evaluate on the complement of idx_noisy_test

    Returns
    -------
    pandas.DataFrame with columns:
        alpha, coverage, sharpness, interval score, n_points
    """
    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    # ---- 1) Train on all available data ----
    X_all = torch.cat([X_train, X_cal], dim=0)
    Y_all = torch.cat([Y_train, Y_cal], dim=0)

    if hasattr(uqmodel, "fit") and callable(uqmodel.fit):
        try:
            uqmodel.fit(X_all, Y_all)
        except TypeError:
            # Fallback for models expecting split args
            uqmodel.fit(X_train, Y_train, X_cal, Y_cal)

    # ---- 2) Normalize indices / mask once ----
    N = X_test.shape[0]
    device = X_test.device

    def _to_index_tensor(idx, N, device):
        """Convert various index types to a 1D LongTensor of indices on `device`."""
        if isinstance(idx, slice):
            return torch.arange(N, device=device, dtype=torch.long)[idx]

        if torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                if idx.numel() != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                return torch.nonzero(idx.to(device), as_tuple=False).squeeze(1).to(torch.long)
            return idx.to(device=device, dtype=torch.long).view(-1)

        if isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)

        if isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                if idx.size != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                mask = torch.from_numpy(idx).to(device=device)
                return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.long)
            return torch.from_numpy(idx).to(device=device, dtype=torch.long).view(-1)

        # Fallback try
        return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

    idx_noise = _to_index_tensor(idx_noisy_test, N, device)

    if compt_smooth:
        # Complement of idx_noise
        mask = torch.ones(N, dtype=torch.bool, device=device)
        if idx_noise.numel() > 0:
            mask[idx_noise] = False
        idx_eval = torch.nonzero(mask, as_tuple=False).squeeze(1)
    else:
        idx_eval = idx_noise

    n_sel = int(idx_eval.numel())

    # Pre-slice Y (used in metrics)
    Y_sel = Y_test[idx_eval] if n_sel > 0 else Y_test[:0]  # empty view when needed

    # ---- 3) Evaluate metrics over selected region ----
    results = []
    for alpha in tqdm(alphas, desc="CP (noisy/complement) grid"):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1) for CP.")

        lower, upper = uqmodel.predict(
            alpha_val, X_test, X_train, Y_train, X_cal, Y_cal,
            heuristic_u=heuristic_u, k=k
        )

        if n_sel == 0:
            coverage = float("nan")
            sharpness = float("nan")
            interval_score = float("nan")
        else:
            pred_sel = (lower[idx_eval], upper[idx_eval])
            coverage = _coverage(pred_sel, Y_sel)
            sharpness = _sharpness(pred_sel)
            interval_score = _interval_score(pred_sel, Y_sel, alpha_val)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sharpness,
            "interval score": interval_score,
            "n_points": n_sel
        })

    return pd.DataFrame(results)



def adaptive_cp_test_uncertainties_grid(
    base_md,                     # a *trained* torch.nn.Module (base predictor)
    alphas,                      # iterable of α values (0<α<1)
    X_train, Y_train,            # training set (for distance / heuristic computations)
    X_cal,   Y_cal,              # calibration set (used to train the quantile net)
    X_test,  Y_test,             # held-out set for evaluation
    *,
    heuristic_u="feature",       # 'feature' | 'latent' | 'raw_std' (passed to AdaptiveCP)
    k=20,                        # k-NN size for heuristics
    conf_nn_hidden_layers=(16, 32, 64, 128, 128, 64, 32, 16), 
    conf_nn_lr=5e-4, conf_nn_epochs=20000,
    idx_noisy_test=None,          # Optional: indices into X_test/Y_test to define a region
    compt_smooth=False,           # NEW: when True, evaluate on the complement of idx_noisy_test
    training_kwargs={"step_size":5000, "gamma":0.5},
    quant_seed=11
):
    """
    For *each* α:
      • instantiates AdaptiveCP(alpha=α)
      • trains its quantile network using (X_cal, Y_cal)
      • builds prediction intervals on X_test
      • computes coverage / sharpness / interval-score
      • If idx_noisy_test is provided:
          - compt_smooth=False (default): evaluate metrics on idx_noisy_test
          - compt_smooth=True:           evaluate metrics on the complement of idx_noisy_test
        Otherwise evaluate on all X_test.

    Returns
    -------
    pd.DataFrame  with one row per α
    """
    import numpy as np
    import pandas as pd
    import torch

    # --- normalize evaluation indices once (independent of alpha loop) ---
    N = X_test.shape[0]
    device = X_test.device

    def _to_index_tensor(idx, N, device):
        """Normalize various index formats to a 1D LongTensor of indices on `device`."""
        if idx is None:
            return None

        if isinstance(idx, slice):
            return torch.arange(N, device=device)[idx].to(torch.long)

        if torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                if idx.numel() != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                return torch.nonzero(idx.to(device), as_tuple=False).squeeze(1).to(torch.long)
            return idx.to(device=device, dtype=torch.long).view(-1)

        if isinstance(idx, (list, tuple)):
            return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

        if isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                if idx.size != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                mask = torch.from_numpy(idx).to(device=device)
                return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.long)
            return torch.from_numpy(idx).to(device=device, dtype=torch.long).view(-1)

        # Fallback: try to tensor-ize
        return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

    idx_eval = None
    if idx_noisy_test is not None:
        idx_noise = _to_index_tensor(idx_noisy_test, N, device)
        if compt_smooth:
            # complement set
            mask = torch.ones(N, dtype=torch.bool, device=device)
            if idx_noise.numel() > 0:
                mask[idx_noise] = False
            idx_eval = torch.nonzero(mask, as_tuple=False).squeeze(1)
        else:
            idx_eval = idx_noise

    # Determine number of points used for metrics
    use_subset = idx_eval is not None
    n_points = int(idx_eval.numel()) if use_subset else int(N)

    results = []

    for alpha in tqdm(alphas, desc="Adaptive-CP grid"):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1).")

        # 1) fresh AdaptiveCP for this alpha
        acp = AdaptiveCP(
            base_md,
            alpha=alpha_val,
            heuristic=heuristic_u,
            conf_nn_hidden_layers=conf_nn_hidden_layers,
            conf_nn_lr=conf_nn_lr,
            conf_nn_epochs=conf_nn_epochs,
            training_kwargs=training_kwargs,
            quant_seed=quant_seed
        )

        # 2) Build intervals on the test set
        lower, upper = acp.predict(
            alpha_val, X_test,
            X_train=X_train, Y_train=Y_train,
            X_cal=X_cal,     Y_cal=Y_cal,
            heuristic_u=heuristic_u, k=k
        )

        # 3) Slice if evaluating only a subset (noisy region or its complement)
        if use_subset:
            pred_set = (lower[idx_eval], upper[idx_eval])
            Y_eval = Y_test[idx_eval]
        else:
            pred_set = (lower, upper)
            Y_eval = Y_test

        # 4) Metrics (guard against empty subset)
        if n_points == 0:
            coverage = float("nan")
            sharpness = float("nan")
            interval_score = float("nan")
        else:
            coverage = _coverage(pred_set, Y_eval)
            sharpness = _sharpness(pred_set)
            interval_score = _interval_score(pred_set, Y_eval, alpha_val)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sharpness,
            "interval score": interval_score,
            "n_points": n_points
        })

    return pd.DataFrame(results)



def adaptive_cp_test_uncertainties_grid_2d_ac(
    base_md,                     # a *trained* torch.nn.Module (base predictor)
    alphas,                      # iterable of α values (0<α<1)
    X_train, Y_train,            # training set (for distance / heuristic computations)
    X_cal,   Y_cal,              # calibration set (used to train the quantile net)
    X_test,  Y_test,             # held-out set for evaluation
    *,
    heuristic_u="feature",       # 'feature' | 'latent' | 'raw_std' (passed to AdaptiveCP)
    k=20,                        # k-NN size for heuristics
    conf_nn_hidden_layers=(16, 32, 64, 128, 128, 64, 32, 16), 
    conf_nn_lr=5e-4, conf_nn_epochs=20000,
    idx_noisy_test=None,          # Optional: indices into X_test/Y_test to define a region
    compt_smooth=False,           # NEW: when True, evaluate on the complement of idx_noisy_test
    training_kwargs={"step_size":5000, "gamma":0.5},
    quant_seed=11
):
    """
    For *each* α:
      • instantiates AdaptiveCP(alpha=α)
      • trains its quantile network using (X_cal, Y_cal)
      • builds prediction intervals on X_test
      • computes coverage / sharpness / interval-score
      • If idx_noisy_test is provided:
          - compt_smooth=False (default): evaluate metrics on idx_noisy_test
          - compt_smooth=True:           evaluate metrics on the complement of idx_noisy_test
        Otherwise evaluate on all X_test.

    Returns
    -------
    pd.DataFrame  with one row per α
    """
    import numpy as np
    import pandas as pd
    import torch

    # --- normalize evaluation indices once (independent of alpha loop) ---
    N = X_test.shape[0]
    device = X_test.device

    def _to_index_tensor(idx, N, device):
        """Normalize various index formats to a 1D LongTensor of indices on `device`."""
        if idx is None:
            return None

        if isinstance(idx, slice):
            return torch.arange(N, device=device)[idx].to(torch.long)

        if torch.is_tensor(idx):
            if idx.dtype == torch.bool:
                if idx.numel() != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                return torch.nonzero(idx.to(device), as_tuple=False).squeeze(1).to(torch.long)
            return idx.to(device=device, dtype=torch.long).view(-1)

        if isinstance(idx, (list, tuple)):
            return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

        if isinstance(idx, np.ndarray):
            if idx.dtype == np.bool_:
                if idx.size != N:
                    raise ValueError("Boolean mask length must match X_test length.")
                mask = torch.from_numpy(idx).to(device=device)
                return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.long)
            return torch.from_numpy(idx).to(device=device, dtype=torch.long).view(-1)

        # Fallback: try to tensor-ize
        return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

    idx_eval = None
    if idx_noisy_test is not None:
        idx_noise = _to_index_tensor(idx_noisy_test, N, device)
        if compt_smooth:
            # complement set
            mask = torch.ones(N, dtype=torch.bool, device=device)
            if idx_noise.numel() > 0:
                mask[idx_noise] = False
            idx_eval = torch.nonzero(mask, as_tuple=False).squeeze(1)
        else:
            idx_eval = idx_noise

    # Determine number of points used for metrics
    use_subset = idx_eval is not None
    n_points = int(idx_eval.numel()) if use_subset else int(N)

    results = []

    for alpha in tqdm(alphas, desc="Adaptive-CP grid"):
        alpha_val = float(alpha)
        if not (0.0 < alpha_val < 1.0):
            raise ValueError("alpha must be in (0,1).")

        # 1) fresh AdaptiveCP for this alpha
        acp = AdaptiveCP_f(
            base_md,
            alpha=alpha_val,
            heuristic=heuristic_u,
            conf_nn_hidden_layers=conf_nn_hidden_layers,
            conf_nn_lr=conf_nn_lr,
            conf_nn_epochs=conf_nn_epochs,
            training_kwargs=training_kwargs,
            quant_seed=quant_seed
        )

        # 2) Build intervals on the test set
        lower, upper = acp.predict(
            alpha_val, X_test,
            X_train=X_train, Y_train=Y_train,
            X_cal=X_cal,     Y_cal=Y_cal,
            heuristic_u=heuristic_u, k=k
        )

        # 3) Slice if evaluating only a subset (noisy region or its complement)
        if use_subset:
            pred_set = (lower[idx_eval], upper[idx_eval])
            Y_eval = Y_test[idx_eval]
        else:
            pred_set = (lower, upper)
            Y_eval = Y_test

        # 4) Metrics (guard against empty subset)
        if n_points == 0:
            coverage = float("nan")
            sharpness = float("nan")
            interval_score = float("nan")
        else:
            coverage = _coverage(pred_set, Y_eval)
            sharpness = _sharpness(pred_set)
            interval_score = _interval_score(pred_set, Y_eval, alpha_val)

        results.append({
            "alpha": alpha_val,
            "coverage": coverage,
            "sharpness": sharpness,
            "interval score": interval_score,
            "n_points": n_points
        })

    return pd.DataFrame(results)
