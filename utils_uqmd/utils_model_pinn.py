from utils_uqmd.interface_model import BasePINNModel

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear


if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon GPU (M1/M2/M3)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # Fallback to CPU

device = torch.device("cpu")
torch.set_default_device(device)
print(f"Using device: {device}")


class DeterministicFeedForwardNN(BasePINNModel):
    """Feed-forward neural network with Deterministic linear layers (for VI)."""
    def __init__(self, input_dim, hidden_dims, output_dim, act_func=nn.Tanh()):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        prev_dim = input_dim
        # Build hidden layers with BayesianLinear
        for h in hidden_dims:
            layers.append(DeterministicLinear(prev_dim, h))  # in_feat, out_feat
            layers.append(act_func)
            prev_dim = h
        # Final output layer (Bayesian linear as well)
        last_layer = DeterministicLinear(prev_dim, output_dim)
        layers.append(last_layer)
        self.layers = nn.ModuleList(layers)  # not using Sequential because it's a mix of custom and activations


    def forward(self, x):
        out = x
        for layer in self.layers:
            # [BL, act, BL, act, ..., act, BL] go through all the layers
            out = layer(out)  # BayesianLinear or activation
        return out


class PINN(DeterministicFeedForwardNN):
    """Learn a PINN model for different 1d PDE"""

    def __init__(self, pde_class, input_dim, hidden_dims, output_dim, activation= nn.Tanh()):
        super().__init__(input_dim, hidden_dims, output_dim, activation)
        self.pde = pde_class

    def fit_pinn(self,
        coloc_pt_num,
        X_train, Y_train,
        λ_pde = 1.0, λ_ic = 10.0, λ_bc = 10.0, λ_data = 5.0,
        epochs = 20_000, lr = 3e-3, print_every = 500,
        scheduler_cls = StepLR, scheduler_kwargs = {'step_size': 5000, 'gamma': 0.5},
        stop_schedule = 40000):

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

        for epoch in range(1, epochs + 1):
            opt.zero_grad()

            # Init them as 0
            loss_data = 0
            loss_pde = 0
            loss_bc = 0
            loss_ic = 0

            # Data loss
            Y_pred = self.forward(X_train)
            loss_data = ((Y_pred - Y_train) ** 2).mean()
            loss=λ_data*loss_data
            # PDE residual
            if hasattr(self.pde, 'residual'):
                loss_pde = self.pde.residual(self, coloc_pt_num)
                loss+=λ_pde * loss_pde
            # B.C. conditions
            if hasattr(self.pde, 'boundary_loss'):
                loss_bc = self.pde.boundary_loss(self)
                loss+=λ_bc * loss_bc
            # I.C. conditions
            if hasattr(self.pde, 'ic_loss'):
                loss_ic = self.pde.ic_loss(self)
                loss+=λ_ic * loss_ic
            loss.backward()
            opt.step()


            # Optionally print training progress
            if epoch % print_every == 0 or epoch == 1:
                print(f"ep {epoch:5d} | L={loss:.2e} | pde={loss_pde:.2e}  "
                      f"ic={loss_ic:.2e}  bc={loss_bc:.2e} | lr={opt.param_groups[0]['lr']:.2e} ")

            if epoch <= stop_schedule:
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    elif isinstance(scheduler, StepLR):
                        scheduler.step()

        return {"Data Loss": data_loss_his, "Initial Condition Loss": ic_loss_his,
                "Boundary Condition Loss": bc_loss_his, "PDE Residue Loss": pde_loss_his}