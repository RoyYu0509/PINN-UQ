from utils_layer_DeterministicLinearLayer import DeterministicLinear
from utils_model_pinn import PINN
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from interface_model import BasePINNModel

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_layer_DeterministicLinearLayer import DeterministicLinear

import numpy as np


if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU (M1/M2/M3)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # Fallback to CPU

device = torch.device("cpu")
torch.set_default_device(device)
print(f"Using device: {device}")


class CPPINN(nn.Module):
    def __init__(self, pde_class, input_dim, hidden_dims, output_dim, act_cls=nn.Tanh):
        super().__init__()
        # Ensure hidden_dims is a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.act = act_cls()  # store activation once
        self.input_layer = DeterministicLinear(input_dim, hidden_dims[0])

        # All *intermediate* hidden linear layers
        inner_layers = []
        prev_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            inner_layers.append(DeterministicLinear(prev_dim, h))
            prev_dim = h
        self.hidden_layers = nn.ModuleList(inner_layers)

        # Final output layer
        self.output_layer = DeterministicLinear(prev_dim, output_dim)
        self.pde = pde_class

    def forward(self, x, return_hidden=False):
        x = self.act(self.input_layer(x))  # first layer + act

        for layer in self.hidden_layers:  # all remaining hidden layers
            x = self.act(layer(x))

        hidden = x  # last hidden representation
        out = self.output_layer(hidden)  # logits / regression output

        return (out, hidden) if return_hidden else out



    def fit_cp_pinn(self,
                 coloc_pt_num,
                 X_train, Y_train,
                 λ_pde=1.0, λ_ic=10.0, λ_bc=10.0, λ_data=5.0,
                 epochs=20_000, lr=3e-3, print_every=500,
                 scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
                 stop_schedule=40000):

        # move model to device
        self.to(device)
        # Optimizer
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # Scheduler
        scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        # Training History
        pde_loss_his = []
        bc_loss_his = []
        ic_loss_his = []
        data_loss_his = []

        for ep in range(1, epochs + 1):
            opt.zero_grad()

            # Init them as 0
            loss_data = 0
            loss_pde = 0
            loss_bc = 0
            loss_ic = 0

            # Data loss
            Y_pred = self.forward(X_train)
            loss_data = ((Y_pred - Y_train) ** 2).mean()
            loss = λ_data * loss_data
            # PDE residual
            if hasattr(self.pde, 'residual'):
                loss_pde = self.pde.residual(self, coloc_pt_num)
                loss += λ_pde * loss_pde
            # B.C. conditions
            if hasattr(self.pde, 'boundary_loss'):
                loss_bc = self.pde.boundary_loss(self)
                loss += λ_bc * loss_bc
            # I.C. conditions
            if hasattr(self.pde, 'ic_loss'):
                loss_ic = self.pde.ic_loss(self)
                loss += λ_ic * loss_ic
            loss.backward()
            opt.step()

            if ep <= stop_schedule:  # Stop decreasing the learning rate
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    elif isinstance(scheduler, StepLR):
                        scheduler.step()

            if (ep % print_every == 0 or ep == 1):  # Only start reporting after the warm-up Phase
                print(f"ep {ep:5d} | L={loss:.2e} | "
                      f"data={loss_data:.2e} | pde={loss_pde:.2e}  "
                      f"ic={loss_ic:.2e}  bc={loss_bc:.2e} | lr={opt.param_groups[0]['lr']:.2e}")

                pde_loss_his.append(loss_pde.item())
                bc_loss_his.append(loss_bc.item())
                data_loss_his.append(loss_data.item())

        return data_loss_his, ic_loss_his, bc_loss_his, pde_loss_his

    ######################## Feature Space Distance ############################
    # Compute the uncertainty as distance in the original feature space
    def _feature_distance(self, X_cal, X_train, k):
        # Use sklearn to compute kNN distances in the feature space
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)
        distances, _ = nbrs.kneighbors(X_cal)

        avg_dists = np.mean(distances, axis=1)
        return avg_dists  # 每一个 calibration point 的 heuristic uncertainty

    def _conf_metric_feature(self, X_cal, Y_cal, X_train, k=10, eps=1e-8):
        """
        Scaled residual score  s_i = |y_i − ŷ_i| / d_i   (vector-valued)

        Parameters:
            - X_cal, Y_cal, X_train: the training data
            - k: the number nearest neighbour.
            - eps: the lower bound to prevent if the calibration point
                being too close to its neibouring training data point, leading
                to blowing up conformal score.
        """
        X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32, device=device)

        # ---> remove the stray [0] so we predict *all* N calibration points
        with torch.no_grad():
            Y_pred = self(X_cal_tensor).cpu().numpy()  # shape (N, out_dim)

        raw_residual = np.abs(Y_cal - Y_pred)  # (N, out_dim)

        # k-NN distance in the *feature* (input) space – unchanged
        latent_dist = self._feature_distance(X_cal, X_train, k=k)  # (N,)

        # ---> prevent divide-by-zero / tiny-distance conformal score blow-ups
        latent_dist = np.maximum(latent_dist, eps)

        scaled_score = raw_residual / latent_dist[:, None]  # (N, out_dim)
        return scaled_score, latent_dist

    ##############################################################################

    ######################### Latent Space Distance ##############################
    def _latent_distance(self, X_cal, X_train, k):
        """
        Compute the average distance from each calibration point to its k nearest neighbors
        from the training set in the latent space.
        """
        self.eval()
        with torch.no_grad():
            H_cal = self(torch.tensor(X_cal, dtype=torch.float32).to(device), return_hidden=True)[
                1].clone().detach().cpu().numpy()
            H_train = self(torch.tensor(X_train, dtype=torch.float32).to(device), return_hidden=True)[
                1].clone().detach().cpu().numpy()

        # Use sklearn to compute kNN distances in latent space
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(H_train)
        distances, _ = nbrs.kneighbors(H_cal)

        avg_dists = np.mean(distances, axis=1)
        return avg_dists  # 每一个 calibration point 的 heuristic uncertainty

    def _conf_metric_latent(self, X_cal, Y_cal, X_train, k=10, eps=1e-8):
        """
        Scaled residual score  s_i = |y_i − ŷ_i| / d_i   (vector-valued)

        Parameters:
            - X_cal, Y_cal, X_train: the training data
            - k: the number nearest neighbour.
            - eps: the lower bound to prevent if the calibration point
                being too close to its neibouring training data point, leading
                to blowing up conformal score.
        """
        X_cal_tensor = torch.tensor(X_cal, dtype=torch.float32, device=device)

        # ---> remove the stray [0] so we predict *all* N calibration points
        with torch.no_grad():
            Y_pred = self(X_cal_tensor).cpu().numpy()  # shape (N, out_dim)

        raw_residual = np.abs(Y_cal - Y_pred)  # (N, out_dim)

        # k-NN distance in the *feature* (input) space – unchanged
        latent_dist = self._latent_distance(X_cal, X_train, k=k)  # (N,)

        # ---> prevent divide-by-zero / tiny-distance conformal score blow-ups
        latent_dist = np.maximum(latent_dist, eps)

        scaled_score = raw_residual / latent_dist[:, None]  # (N, out_dim)
        return scaled_score, latent_dist

    ##############################################################################

    def predict(self, alpha, k, X_test, Y_test, X_cal, Y_cal, X_train, distance_space):
        """
        Implements conformal prediction with latent-space-scaled residuals
        """
        self.k = k
        n = len(X_cal)

        if distance_space == "feature":
            conf_metric_func = self._conf_metric_feature
            distance_func = self._feature_distance
        elif distance_space == "latent":
            conf_metric_func = self._conf_metric_latent
            distance_func = self._latent_distance

        # Step 1: Scaled scores on calibration set
        cal_scores, _ = conf_metric_func(X_cal, Y_cal, X_train, k=self.k)
        qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, axis=0, method="higher")

        # Step 2: Predict and get latent distances for test set
        with torch.no_grad():
            y_pred_test, _ = self(torch.tensor(X_test, dtype=torch.float32).to(device), return_hidden=True)
            y_pred_test = y_pred_test.cpu().numpy()
            test_dists = distance_func(X_test, X_train, k)

        # Step 3: Prediction intervals
        eps = qhat * test_dists[:, None]  # shape: [n_test, output_dim]
        lower = y_pred_test - eps
        upper = y_pred_test + eps
        lower = torch.tensor(lower, dtype=torch.float32).to(device)
        upper = torch.tensor(upper, dtype=torch.float32).to(device)

        return [lower, upper]

    #####################################################################
    def naive_predict(self, k, X_test, Y_test, X_train, Y_train, device="cpu", factor: float = 3.0):
        """
        Heuristic UQ via k-NN sample variance.

        Parameters
        ----------
        X_test, Y_test : np.ndarray  – test inputs / true targets
        X_train, Y_train : np.ndarray  – training inputs / targets
        factor : float  – scale-up factor for the error band (default = 1.0)

        Returns
        -------
        bounds : [lower, upper]  – arrays with the same shape as Y_test
        coverage : float         – mean test-set coverage of the interval
        """

        # ----- 1. model mean prediction on the test set -----
        self.eval()
        with torch.no_grad():
            y_pred = self(
                torch.as_tensor(X_test, dtype=torch.float32, device=device)
            ).cpu().numpy()  # shape (n_test, out_dim)

        # ----- 2. k-nearest neighbours in input space -----
        k = k
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_train)
        _, idx = nbrs.kneighbors(X_test)  # idx: (n_test, k)

        # ----- 3. local sample variance of neighbour targets -----
        neigh_targets = Y_train[idx]  # (n_test, k, out_dim)
        # unbiased variance; fall back to 0 when k == 1
        var_local = np.var(neigh_targets, axis=1, ddof=1 if k > 1 else 0)
        sigma_local = np.sqrt(var_local) * factor  # (n_test, out_dim)

        # ----- 4. prediction bands -----
        lower = y_pred - sigma_local
        upper = y_pred + sigma_local

        # ----- 5. empirical coverage on the held-out set -----
        coverage = np.mean((Y_test >= lower) & (Y_test <= upper))

        return [lower, upper], coverage

