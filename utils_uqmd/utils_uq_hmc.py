# utils_uq_hmc_fast.py   ──  v3  (tqdm + safe grad on mps/cpu/cuda)
# -----------------------------------------------------------------------------
import torch, math
import torch.nn as nn
from torch.func import functional_call          # still used for fast forward
from tqdm.auto import trange                    # progress-bars

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params])

def unflatten_params(model, theta_vec):
    """Write theta_vec back into model.parameters(), in-place."""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data = theta_vec[idx:idx+n].view_as(p)
        idx += n
    assert idx == theta_vec.numel(), "θ size mismatch"

# -----------------------------------------------------------------------------
# FastHMCBPINN
# -----------------------------------------------------------------------------
class HMCBPINN(nn.Module):
    """
    Vectorised HMC (multi-chain) PINN with tqdm bars and device-safe grads.
    """
    def __init__(self, pde_class, input_dim, hidden_dims, output_dim,
                 act_func=nn.Tanh, prior_std=1.0,
                 step_size=1e-3, leapfrog_steps=10, chains=1, device='cpu'):
        super().__init__()
        self.pde        = pde_class
        self.prior_std2 = prior_std**2
        self.eps        = step_size
        self.L          = leapfrog_steps
        self.chains     = chains
        self.device     = device

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != output_dim:
                layers.append(act_func())
        self.net = nn.Sequential(*layers).to(device)

        self._posterior = []
        self._coloc_pt_num = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, params=None):
        if params is None:
            return self.net(x)
        return functional_call(self.net, params, (x,))

    # ------------------------------------------------------------------
    # log-prior & likelihood
    # ------------------------------------------------------------------
    def _log_prior(self, theta):
        return -0.5 * theta.pow(2).sum() / self.prior_std2

    def _pde_losses(self, coloc_pt_num):
        pde = ic = bc = torch.tensor(0., device=self.device)
        if hasattr(self.pde, "residual") and coloc_pt_num is not None:
            pde = self.pde.residual(self, coloc_pt_num)
        if hasattr(self.pde, "boundary_loss"):
            bc = self.pde.boundary_loss(self)
        if hasattr(self.pde, "ic_loss"):
            ic = self.pde.ic_loss(self)
        return pde, ic, bc

    def _log_likelihood(self, X, Y, lam_pde, lam_ic, lam_bc, lam_data):
        data_loss = torch.nn.functional.mse_loss(self(X), Y)
        pde_loss, ic_loss, bc_loss = self._pde_losses(self._coloc_pt_num)
        return -(lam_pde*pde_loss + lam_ic*ic_loss +
                 lam_bc*bc_loss + lam_data*data_loss), \
               {"Data": data_loss, "PDE": pde_loss,
                "IC": ic_loss, "BC": bc_loss}

    # ------------------------------------------------------------------
    # fit  (MAP + HMC)
    # ------------------------------------------------------------------
    def fit(self, coloc_pt_num, X_train, Y_train,
            lam_pde=1., lam_ic=0., lam_bc=1., lam_data=1.,
            epochs=5000, lr=1e-3,
            hmc_samples=2000, burn_in=500,
            print_every=500, step_size=None, leapfrog_steps=None, **unused):
        
        if step_size is not None:
            self.eps = step_size
        if leapfrog_steps is not None:
            self.L = leapfrog_steps

        self._coloc_pt_num = coloc_pt_num
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

        opt  = torch.optim.Adam(self.parameters(), lr=lr)
        hist = {k: [] for k in ("Total", "Data", "PDE", "IC", "BC")}

        # -------- 1️⃣ MAP optimisation ----------------------------------
        map_bar = trange(1, epochs+1, desc="MAP", leave=False)
        for ep in map_bar:
            opt.zero_grad()
            logL, parts = self._log_likelihood(X_train, Y_train,
                                               lam_pde, lam_ic, lam_bc, lam_data)
            nlp = -(self._log_prior(flatten_params(self.parameters())) + logL)
            nlp.backward(); opt.step()

            hist["Total"].append(nlp.item())
            for k,v in parts.items(): hist[k].append(v.item())

            if ep % print_every == 0 or ep == epochs:
                map_bar.write(
                    f"[MAP] epoch {ep:6d}  −logPost={nlp.item():.3e}  "
                    + "  ".join(f"{k}={parts[k]:.3e}" for k in parts)
                )
            map_bar.set_postfix(loss=f"{nlp.item():.2e}")

        # -------- prepare θ0 for chains --------------------------------
        θ0 = flatten_params(self.parameters()).detach().to(self.device)
        θ0 = θ0.repeat(self.chains, 1)                     # (B,D)

        # potential energy
        def U_scalar(theta_vec):
            unflatten_params(self, theta_vec)
            logL, _ = self._log_likelihood(X_train, Y_train,
                                           lam_pde, lam_ic, lam_bc, lam_data)
            return -(self._log_prior(theta_vec) + logL)

        def grad_U(theta_mat):                     # (B,D) → (B,D)
            grads = []
            for theta in theta_mat:                # loop over chains
                theta = theta.detach().clone().requires_grad_(True)

                # write the *views* of theta into the model — safe for autograd
                unflatten_params(self, theta)

                logL, _ = self._log_likelihood(X_train, Y_train,
                                            lam_pde, lam_ic, lam_bc, lam_data)
                U       = -(self._log_prior(theta) + logL)

                g, = torch.autograd.grad(U, theta, retain_graph=False)
                grads.append(g.detach())
            return torch.stack(grads, 0)


        accept_cnt = 0
        θ = θ0.clone()
        # print(f"Using step size: {self.eps}, Using Length: {self.L}")
        g = grad_U(θ)
        # print(f"∥∇U∥ (mean across chains): {g.norm(dim=1).mean().item():.3e}")

        # -------- 2️⃣ HMC sampling ------------------------------------
        hmc_bar = trange(1, hmc_samples+1, desc="HMC", leave=False)
        for it in hmc_bar:
            p = torch.randn_like(θ)
            θ_curr, p_curr = θ.clone(), p.clone()

            # half-step momentum
            p = p - 0.5*self.eps*grad_U(θ)

            θ_before = θ.clone()

            for _ in range(self.L):
                θ = θ + self.eps*p
                p = p - self.eps*grad_U(θ)
            p = p - 0.5*self.eps*grad_U(θ)
            p = -p                                          # Negate momentum
            delta_theta = (θ - θ_before).norm(dim=1)
            # print(f"Δθ (per chain): {delta_theta}")

            # energies
            U_curr = torch.stack([U_scalar(row) for row in θ_curr])
            U_prop = torch.stack([U_scalar(row) for row in θ])
            K_curr = 0.5*p_curr.pow(2).sum(dim=1)
            K_prop = 0.5*p.pow(2).sum(dim=1)

            delta_H = (U_prop + K_prop) - (U_curr + K_curr)
            acc_prob = torch.exp(-delta_H).clamp(max=1.0)
            accept = (torch.rand_like(acc_prob) < acc_prob).view(-1, 1)
            θ        = torch.where(accept, θ, θ_curr)
            accept_cnt += accept.float().sum().item()

            if it > burn_in:
                self._posterior.append(θ.clone().cpu())

            if it % print_every == 0 or it == hmc_samples:
                hmc_bar.write(f"[HMC] iter {it:6d}  acc-rate={accept_cnt/it:.2f}")
            
            hmc_bar.set_postfix(acc=f"{accept_cnt/it:.2f}")

        print(f"Finished HMC: avg acceptance {accept_cnt / (hmc_samples * self.chains):.3f}")
        print(f"Keep {len(self._posterior)} posterior sample from the HMC algo")
        unflatten_params(self, θ[0])           # restore weights
        return hist

    @torch.no_grad()
    def predict(self, alpha, X_test, use_chain=0,  **unused):
        """Draw samples from the HMC posterior and return prediction bounds
        with configurable confidence level."""

        self.eval()
        X_test = X_test.to(self.device)
        preds = []

        for θ in self._posterior:
            unflatten_params(self, θ[use_chain].to(self.device))
            y_pred = self(X_test)
            preds.append(y_pred.detach())

        preds = torch.stack(preds)  # shape: [n_samples, batch_size, output_dim]
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)

        # Convert alpha value to z_score
        alpha_tensor = torch.tensor([alpha / 2], device=X_test.device, dtype=torch.float32)
        z_score = torch.distributions.Normal(0, 1).icdf(1 - alpha_tensor).abs().item()

        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std

        return (lower_bound, upper_bound)

    

    # ------------------------------------------------------------------
    # evaluation: test MSE
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss (MSE) on the given dataset."""
        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)
        preds  = self(X_test)
        loss   = torch.nn.functional.mse_loss(preds, Y_test, reduction="mean")
        return loss.item()
