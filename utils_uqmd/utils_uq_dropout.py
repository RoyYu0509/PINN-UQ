import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils_uqmd.utils_model_pinn import *
from utils_uqmd.utils_layer_DeterministicLinearLayer import DeterministicLinear

# ───────────────────────────────────────────────────────────────────────────────
#  1.  Feed-forward network that inserts Drop-out after every deterministic layer
# ───────────────────────────────────────────────────────────────────────────────

class DeterministicDropoutNN(nn.Module):
    """
    Fully–connected network identical to DeterministicFeedForwardNN,
    but each hidden block is Linear ▸ Dropout ▸ Activation.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float,
                 act_func = nn.Tanh()
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.p_drop = p_drop
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                DeterministicLinear(prev, h),
                nn.Dropout(p_drop),           # <── new
                act_func
            ])
            prev = h
        layers.append(DeterministicLinear(prev, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ───────────────────────────────────────────────────────────────────────────────
#  2.  PINN wrapper with the standard physics losses *plus* MC-Drop-out UQ
# ───────────────────────────────────────────────────────────────────────────────

class DropoutPINN(nn.Module):
    """
    Physics-Informed Neural Network with Monte-Carlo Drop-out for UQ.

    Usage
    -----

    """
    def __init__(self,
                 pde_class,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float = 0.1,
                 activation = nn.Tanh()):
        super().__init__()
        # Pure neural net part
        self.net = DeterministicDropoutNN(input_dim, hidden_dims,
                                          output_dim, p_drop, activation)
        self.pde = pde_class               # physics callbacks

    # Register backbone as a sub-module (it already is because of the attr)
    def forward(self, x):
        return self.net(x)

    def fit(self,
        # ------------ args ----------------
        coloc_pt_num,
        X_train, Y_train,
        # ----------- kwargs ---------------
        λ_pde=1.0, λ_ic=10.0, λ_bc=10.0, λ_data=5.0,
        epochs=20_000, lr=3e-3, print_every=500,
        scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
        stop_schedule=40000
        ):
        
        device = X_train.device
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

        self.train()
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

    # -------------------------------------------------------------------------
    # Uncertainty-aware prediction
    # -------------------------------------------------------------------------
    # Dropout Model
    @torch.inference_mode()
    def predict(
        # ------------ args ---------------
        self, alpha,
        X_test,  
        # ----------- kwargs ---------------
        n_samples: int = 100,
        keep_dropout: bool = True
    ):
        if keep_dropout:
            self.train()
            self.enable_mc_dropout()

        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(X_test))
        preds = torch.stack(preds)                         # (S, N, out)
        mean = preds.mean(0)
        std  = preds.std(0)

        # Two-sided (1-alpha) Gaussian interval
        z = torch.tensor(
            abs(torch.distributions.Normal(0,1).icdf(torch.tensor(alpha/2))),
            device=preds.device, dtype=preds.dtype
        )
        lower = mean - z*std
        upper = mean + z*std

        return (lower, upper)

    # -------------------------------------------------------------------------
    # Helper: keep dropout layers “on” during evaluation
    # -------------------------------------------------------------------------
    def enable_mc_dropout(self):
        """
        Puts *all* Dropout sub-modules into training mode while leaving the rest
        of the network untouched.  Recommended before calling `predict_uq`.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss on the testing data set"""

        preds = self(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean")
        # If the caller asked for a reduced value, return the Python float
        return loss.item() 