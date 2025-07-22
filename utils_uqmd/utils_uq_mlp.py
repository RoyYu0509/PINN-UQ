from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear
from utils_uqmd.utils_model_pinn import PINN
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from utils_uqmd.interface_model import BasePINNModel

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear

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


class MLPPINN(nn.Module):
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
        x = self.act(self.input_layer.forward(x))  # first layer + act

        for layer in self.hidden_layers:  # all remaining hidden layers
            x = self.act(layer(x))

        hidden = x  # last hidden representation
        out = self.output_layer.forward(hidden)  # logits / regression output

        return (out, hidden) if return_hidden else out


    # MLP Model
    def fit(self,
        # ------------ args ----------------
        coloc_pt_num,
        X_train, Y_train,
        # ----------- kwargs ---------------
        λ_pde=1.0, λ_ic=5.0, λ_bc=5.0, λ_data=1.0,
        epochs=20_000, lr=3e-3, print_every=500,
        scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
        stop_schedule=40000
    ):

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
            data_loss_his.append(loss_data.item())

            # PDE residual
            if hasattr(self.pde, 'residual'):
                loss_pde = self.pde.residual(self, coloc_pt_num)
                loss += λ_pde * loss_pde
                pde_loss_his.append(loss_pde.item())
            # B.C. conditions
            if hasattr(self.pde, 'boundary_loss'):
                loss_bc = self.pde.boundary_loss(self)
                loss += λ_bc * loss_bc

                bc_loss_his.append(loss_bc.item())

            # I.C. conditions
            if hasattr(self.pde, 'ic_loss'):
                loss_ic = self.pde.ic_loss(self)
                loss += λ_ic * loss_ic

                ic_loss_his.append(loss_ic.item())
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

        return {"Data": data_loss_his, "Initial Condition Loss": ic_loss_his,
                "Boundary Condition Loss": bc_loss_his, "PDE Residue Loss": pde_loss_his}

    # MLP Model
    def predict(
        # ------------ args ---------------
        self, alpha,
        X_test,  
        # ----------- kwargs ---------------
    ):
        """Return the predcited result with no uq"""
        mean = self.forward(X_test)
        lower = torch.tensor(mean, device=X_test.device)
        upper = torch.tensor(mean, device=X_test.device)
        return (lower, upper)
    
    #####################################################################
    def local_var_predict(self, alpha, X_test, X_train, Y_train, k=10,device="cpu"):
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
        """

        # ----- 1. model mean prediction on the test set -----
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(
                torch.as_tensor(X_test, dtype=torch.float32, device=device)
            ).cpu().numpy()  # shape (n_test, out_dim)

        # ----- 2. k-nearest neighbours in input space -----
        k = k
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X_train)
        _, idx = nbrs.kneighbors(X_test)  # idx: (n_test, k)

        # ----- 3. local sample variance of neighbour targets -----
        neigh_targets = Y_train[idx]  # (n_test, k, out_dim)
        # unbiased variance; fall back to 0 when k == 1
        var_local = np.var(neigh_targets.cpu().numpy(), axis=1, ddof=1 if k > 1 else 0)
        sigma_local = np.sqrt(var_local)  # (n_test, out_dim)

        # ----- 4. prediction bands -----
        z = torch.tensor(
            abs(torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha / 2)))
        ).cpu().numpy()
        lower = y_pred - sigma_local * z
        upper = y_pred + sigma_local * z

        lower = torch.tensor(lower, device=X_test.device)
        upper = torch.tensor(upper, device=X_test.device)
        return [lower, upper]

    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss on the testing data set"""
        preds = self(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean")
        # If the caller asked for a reduced value, return the Python float
        return loss.item() 