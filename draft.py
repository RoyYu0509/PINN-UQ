from interface_model import BasePINNModel
from utils_layer_BayesianLinearLayer import BayesianLinearLayer as BayesianLinear
from utils_model_bpinn import BayesianFeedForwardNN

import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau



class VIBPINN(BayesianFeedForwardNN):
    """Bayesian PINN training using Variational Inference (mean-field Gaussian approximation)."""
    def __init__(self, pde_class, input_dim, hidden_dims, output_dim, mu_std, rho, prior_std=1.0, act_func=nn.Tanh()):
        """
        model: BayesianFeedForwardNN (with BayesianLinear layers).
        pde: an instance of Poisson1D, DampedOscillator1D, or similar (must have residual() & boundary_loss()).
        x_collocation: tensor of collocation points in domain interior for physics residual.
        x_boundary: tensor of boundary (or initial) points.
        boundary_values: actual values at boundary (if needed by some PDE, not used directly if pde.boundary_loss handles it).
        """
        super().__init__(input_dim, hidden_dims, output_dim, mu_std, rho, prior_std, act_func)
        self.pde = pde_class

    def fit_vi_bpinn(self,
        coloc_pt_num,
        X_train=torch.tensor, Y_train=torch.tensor,
        data_noise_guess = 1.0,
        λ_pde=1.0, λ_ic=1.0, λ_bc=1.0, λ_elbo=1.0, λ_data=1.0,
        epochs=20_000, lr=3e-3,
        scheduler_cls=StepLR, scheduler_kwargs={'step_size': 5000, 'gamma': 0.5},
        stop_schedule=40000):
        # Optimizer
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # Scheduler
        scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        # Training History
        pde_loss_his = []
        bc_loss_his = []
        ic_loss_his = []
        data_loss_his = []


        # Determine what are the losses the pde have
        has_residue_l = hasattr(self.pde, 'residual')
        has_bc_l = hasattr(self.pde, 'boundary_loss')
        has_ic_l = hasattr(self.pde, 'ic_loss')

        # Autogenerate print_every setting
        print_every = epochs / 100

        self.train()

        for epoch in range(epochs):
            opt.zero_grad()

            loss_pde = 0
            loss_bc = 0
            loss_ic = 0

            # Compute physics-informed loss:
            # normalized KL Divergence
            total_pt_num = coloc_pt_num + X_train.shape[0]
            kl_div = self.kl_divergence()/total_pt_num

            # NLL
            Y_pred = self.forward(X_train)
            loss_data = self.nll_gaussian(Y_pred, Y_train, data_noise_guess=data_noise_guess)

            # Negative ELBO
            n_elbo = loss_data+kl_div
            data_loss_his.append(n_elbo)

            # PDE residual
            if has_residue_l:
                loss_pde = (self.pde.residual(self, coloc_pt_num)**2).mean()
                pde_loss_his.append(loss_pde)

            # B.C. conditions
            if has_bc_l:
                loss_bc = self.pde.boundary_loss(self)
                bc_loss_his.append(loss_bc)

            # I.C. conditions
            if has_ic_l:
                loss_ic = self.pde.ic_loss(self)
                ic_loss_his.append(loss_ic)

            # Combined loss = physics loss (acts like negative log-likelihood) + KL/N
            loss = λ_pde*loss_pde +  λ_ic*loss_ic + λ_bc*loss_bc + λ_elbo*n_elbo
            loss.backward()
            opt.step()

            # Optionally print progress
            if epoch % print_every == 0 or epoch == 1:  # Only start reporting after the warm-up Phase
                print(f"ep {epoch:5d} | L={loss:.2e} | "
                      f"elbo={n_elbo:.2e} | pde={loss_pde:.2e}  "
                      f"ic={loss_ic:.2e}  bc={loss_bc:.2e} | lr={opt.param_groups[0]['lr']:.2e}")


            if epoch <= stop_schedule:  # Stop decreasing the learning rate
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    elif isinstance(scheduler, StepLR):
                        scheduler.step()

        return {"ELBO": data_loss_his, "Initial Condition Loss": ic_loss_his, "Boundary Condition Loss": bc_loss_his, "PDE Residue Loss": pde_loss_his}



    def predict(self, x_test, n_samples=100):
        """Draw samples from the variational posterior to estimate predictive mean and variance."""
        self.eval()
        preds = []
        for _ in range(n_samples):
            # Forward pass will sample a new set of weights each time (due to BayesianLinear)
            y_pred = self.forward(x_test)
            preds.append(y_pred.detach())
        preds = torch.stack(preds)  # shape [n_samples, batch_size, output_dim]
        mean = preds.mean(dim=0)
        var = preds.var(dim=0)  # predictive variance
        return mean, var