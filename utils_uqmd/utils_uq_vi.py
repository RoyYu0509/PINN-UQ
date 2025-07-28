from utils_uqmd.interface_model import BasePINNModel
from utils_uqmd.utils_layer_BayesianLinearLayer import BayesianLinearLayer as BayesianLinear
from utils_uqmd.utils_model_bpinn import BayesianFeedForwardNN

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class VIBPINN(BayesianFeedForwardNN):
    """Bayesian PINN training using Variational Inference (mean-field Gaussian approximation), with learnable data noise."""
    def __init__(self, pde_class, input_dim, hidden_dims, output_dim,
                 mu_std = 0.01, rho = -3, prior_std=1.0, init_data_noise=1.0, learn_data_noise=False, act_func=nn.Tanh()):
        """
        pde_class: an instance of a PDE class (e.g., Poisson1D, DampedOscillator1D, etc.)
        input_dim: input dimension size
        hidden_dims: list of hidden layer dimensions
        output_dim: output dimension size
        mu_std, rho: parameters for the Bayesian linear layers
        prior_std: standard deviation of the prior on the weights (default 1.0)
        act_func: activation function (default Tanh)
        init_data_noise: initial guess for the data noise standard deviation (learned during training)
        """
        super().__init__(input_dim, hidden_dims, output_dim, mu_std, rho, prior_std, act_func)
        self.pde = pde_class
        # Define a learnable parameter for the log of the data noise standard deviation.
        if learn_data_noise:
            self.log_noise = nn.Parameter(torch.tensor(math.log(init_data_noise), dtype=torch.float32))
        else:
            self.log_noise = torch.tensor(math.log(init_data_noise), dtype=torch.float32)

    # Variational Inference Model
    def fit(self,
        # ------------ args ----------------
        coloc_pt_num,
        X_train=torch.tensor, Y_train=torch.tensor,
        # ----------- kwargs --------------- 
        λ_pde=1.0, λ_ic=1.0, λ_bc=1.0, λ_elbo=1.0, λ_data=1.0,
        epochs=20_000, lr=3e-3,
        scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
        stop_schedule=40000
    ):

        # Optimizer: note that self.log_noise is included among the parameters.
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # Scheduler
        scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        # Training histories
        pde_loss_his = []
        bc_loss_his = []
        ic_loss_his = []
        nelbo_loss_his = []
        data_loss_his = []

        # Check which losses the PDE provides
        has_residue_l = hasattr(self.pde, 'residual')
        has_bc_l = hasattr(self.pde, 'boundary_loss')
        has_ic_l = hasattr(self.pde, 'ic_loss')

        print_every = epochs / 100

        self.train()

        for epoch in range(epochs):
            opt.zero_grad()

            loss_pde = 0
            loss_bc = 0
            loss_ic = 0

            # Total number of points for normalization
            total_pt_num = coloc_pt_num + X_train.shape[0]
            kl_div = self.kl_divergence() / total_pt_num

            # Compute the predictions and the negative log-likelihood loss,
            # using the learned noise standard deviation: noise_std = exp(log_noise)
            Y_pred = self.forward(X_train)
            noise_std = torch.exp(self.log_noise)
            loss_data = self.nll_gaussian(Y_pred, Y_train, data_noise_guess=noise_std)
            data_loss_his.append(loss_data.item())

            # Negative ELBO
            n_elbo = loss_data + kl_div
            nelbo_loss_his.append(n_elbo.item())
    
            # PDE residual loss (if applicable)
            if has_residue_l:
                loss_pde = (self.pde.residual(self, coloc_pt_num)**2).mean()
                pde_loss_his.append(loss_pde.item())

            # Boundary conditions loss (if applicable)
            if has_bc_l:
                loss_bc = self.pde.boundary_loss(self)
                bc_loss_his.append(loss_bc.item())

            # Initial conditions loss (if applicable)
            if has_ic_l:
                loss_ic = self.pde.ic_loss(self)
                ic_loss_his.append(loss_ic.item())

            # Combined loss: physics loss + negative ELBO + regularization terms
            loss = λ_pde * loss_pde + λ_ic * loss_ic + λ_bc * loss_bc + λ_elbo * n_elbo
            loss.backward()
            opt.step()

            # Optionally print training progress
            if epoch % print_every == 0 or epoch == 1:
                print(f"ep {epoch:5d} | L={loss:.2e} | elbo={n_elbo:.2e} | pde={loss_pde:.2e}  "
                      f"ic={loss_ic:.2e}  bc={loss_bc:.2e} | lr={opt.param_groups[0]['lr']:.2e} "
                      f"| learned noise_std={noise_std.item():.3e}")

            if epoch <= stop_schedule:
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    elif isinstance(scheduler, StepLR):
                        scheduler.step()

        return {"Data":data_loss_his, "ELBO": nelbo_loss_his, "Initial Condition Loss": ic_loss_his,
                "Boundary Condition Loss": bc_loss_his, "PDE Residue Loss": pde_loss_his}

    # Variational Inference
    def predict(
        # ------------ args ---------------
        self, alpha,
        X_test,  
        # ----------- kwargs --------------- 
        n_samples=20000
    ):
        """Draw samples from the variational posterior and return prediction bounds
        with configurable confidence level."""
        self.eval()
        preds = []
        for _ in range(n_samples):
            y_pred = self.forward(X_test)
            preds.append(y_pred.detach())
        preds = torch.stack(preds)

        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        # Convert alpha value to z_score
        alpha_tensor = torch.tensor([alpha / 2], device=X_test.device, dtype=torch.float32)
        z_score = torch.distributions.Normal(0, 1).icdf(1 - alpha_tensor).abs().item()
        

        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std

        return (lower_bound, upper_bound)
    
    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss on the testing data set"""

        preds = self(X_test)
        loss  = torch.nn.functional.mse_loss(preds, Y_test,
                                             reduction="mean")
        # If the caller asked for a reduced value, return the Python float
        return loss.item() 