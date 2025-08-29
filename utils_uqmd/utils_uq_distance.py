# ────────────────────────────────────────────────────────────────────────────────
#  DistanceUQPINN  –  PINN + distance-based uncertainty in one class
# ────────────────────────────────────────────────────────────────────────────────
import torch, math, numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.interface_model import BasePINNModel
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear


class DistanceUQPINN(BasePINNModel):
    """
    A Physics-Informed Neural Network whose prediction **bands** are determined
    purely by k-NN distance in either input (feature) or hidden (latent) space.
    No conformal calibration is used.
    """

    # ───────────────────── constructor ─────────────────────
    def __init__(
        self,
        pde_class,
        input_dim:   int,
        hidden_dims: list | int,
        output_dim:  int,
        activation:  nn.Module = nn.Tanh(),
        device:      torch.device = "cpu"
    ):
        super().__init__()
        self.pde = pde_class
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Build feed-forward backbone
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [DeterministicLinear(prev, h), activation]
            prev = h
        layers += [DeterministicLinear(prev, output_dim)]
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.to(self.device)

        # Training data cache (needed for distance computation later)
        self._X_train_cached = None


    # ───────────────────── forward pass ─────────────────────
    def forward(self, x: torch.Tensor, *, return_hidden: bool = False):
        """
        Parameters
        ----------
        x : (N, input_dim) tensor
        return_hidden : if True, also return last hidden representation
        """
        out, hidden = x, None
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, DeterministicLinear):
                hidden = out          # capture after linear, before activation
        return (out, hidden) if return_hidden else out


    # ───────────────────── trainer ─────────────────────
    def fit(
        self,
        coloc_pt_num:  int,
        X_train:       torch.Tensor,
        Y_train:       torch.Tensor,
        *,
        λ_pde:         float          = 1.0,
        λ_ic:          float          = 10.0,
        λ_bc:          float          = 10.0,
        λ_data:        float          = 5.0,
        epochs:        int            = 20_000,
        lr:            float          = 3e-3,
        print_every:   int            = 500,
        scheduler_cls                 = StepLR,
        scheduler_kwargs: dict        = {'step_size': 5000, 'gamma': 0.7},
        stop_schedule: int            = 40_000
    ):
        """Standard PINN training (MSE + PDE/BC/IC losses)."""
        # Cache training inputs for distance heuristics
        self._X_train_cached = X_train.detach()

        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sched = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        for ep in range(1, epochs + 1):
            opt.zero_grad()
            loss = λ_data * ((self.forward(X_train) - Y_train) ** 2).mean()

            if hasattr(self.pde, 'residual'):
                loss += λ_pde * self.pde.residual(self, coloc_pt_num)
            if hasattr(self.pde, 'boundary_loss'):
                loss += λ_bc * self.pde.boundary_loss(self)
            if hasattr(self.pde, 'ic_loss'):
                loss += λ_ic * self.pde.ic_loss(self)

            loss.backward(); opt.step()

            if ep % print_every == 0 or ep == 1:
                print(f"ep {ep:5d} | L={loss:.2e} | lr={opt.param_groups[0]['lr']:.1e}")

            if ep <= stop_schedule and sched:
                if isinstance(sched, ReduceLROnPlateau):
                    sched.step(loss.item())
                else:
                    sched.step()


    # ───────────────────── distance helpers ─────────────────────
    @staticmethod
    def _to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def _feature_dist(self, X_test: torch.Tensor, k: int):
        """Mean k-NN distance in the raw input space."""
        trn, tst = self._to_np(self._X_train_cached), self._to_np(X_test)
        nnm = NearestNeighbors(n_neighbors=k).fit(trn)
        d, _ = nnm.kneighbors(tst)
        return d.mean(axis=1)                       # (N,)

    def _latent_dist(self, X_test: torch.Tensor, k: int):
        """Mean k-NN distance in the last hidden layer."""
        with torch.no_grad():
            H_trn = self.forward(self._X_train_cached.to(self.device), return_hidden=True)[1]
            H_tst = self.forward(X_test.to(self.device),               return_hidden=True)[1]
        trn, tst = self._to_np(H_trn), self._to_np(H_tst)
        nnm = NearestNeighbors(n_neighbors=k).fit(trn)
        d, _ = nnm.kneighbors(tst)
        return d.mean(axis=1)


    # ───────────────────── prediction with bands ─────────────────────
    def predict(
        self,
        alpha:       float,
        X_test:      torch.Tensor,
        *,
        heuristic_u: str   = "feature",   # 'feature' | 'latent'
        n_samples:           int   = 20,
        return_band: bool  = True,
    ):
        """
        Parameters
        ----------
        X_test : (N, input_dim) tensor
        heuristic_u : 'feature' | 'latent'
        k : int              number of neighbours for k-NN distance
        scale : float        converts distance → predictive σ̂
        return_band : bool   if False, return point predictions only
        alpha : float        mis-coverage; 0.05 ⇒ 95 % interval

        Returns
        -------
        • ŷ                        if return_band is False
        • (lower, upper) tensors    if return_band is True
        """
        X_test = X_test.to(self.device)

        # ─── point prediction ───────────────────────────────────────────
        with torch.no_grad():
            y_pred = self.forward(X_test)          # (N, out_dim)

        if not return_band:
            return y_pred                          # just the mean

        if self._X_train_cached is None:
            raise RuntimeError("Must call fit() before predict() so training data is cached.")

        # ─── distance-based scale → σ̂ ──────────────────────────────────
        dist = (self._feature_dist(X_test, n_samples) if heuristic_u == "feature"
                else self._latent_dist(X_test, n_samples))                     # (N,)
        sigma_hat = torch.from_numpy(dist)[:, None].to(self.device)   # (N,1)

        alpha_tensor = torch.as_tensor(1.0 - alpha / 2.0, device=self.device)
        z = torch.distributions.Normal(0.0, 1.0).icdf(alpha_tensor)

        # ─── (1-α) interval:  ŷ ± z·σ̂ ───────────────────────────────
        lower = y_pred - z * sigma_hat
        upper = y_pred + z * sigma_hat
        return (lower, upper)



    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss on the testing data set"""
        preds = self(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean")
        # If the caller asked for a reduced value, return the Python float
        return loss.item() 
