# utils_uq_hmc_fast.py   ──  v3  (tqdm + safe grad on mps/cpu/cuda)
# -----------------------------------------------------------------------------
import torch, math
import torch.nn as nn
from torch.func import functional_call          # still used for fast forward
from tqdm.auto import trange                    # progress-bars
import copy
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


def xavier_init(layer, gain=None):
    """
    Xavier initialization for nn.Linear layers (good for tanh).
    """
    if gain is None:
        gain = nn.init.calculate_gain("tanh")
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(layer.bias, -bound, bound)


class HMCBPINN(nn.Module):
    """
    Vectorised HMC PINN with tqdm bars and device-safe grads.
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
            linear = nn.Linear(d_in, d_out)
            xavier_init(linear) 
            layers.append(linear)
            if d_out != output_dim:
                layers.append(act_func())
        self.net = nn.Sequential(*layers).to(device)

        self._posterior = []
        self._coloc_pt_num = None
        self._num_params = sum(p.numel() for p in self.parameters())

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
    # fit  (MAP + HMC)          <<<  DROP-IN REPLACEMENT
    # ------------------------------------------------------------------
    def fit(self, coloc_pt_num, X_train, Y_train,
            lam_pde=1., lam_ic=0., lam_bc=1., lam_data=1.,
            epochs=5000, lr=1e-3,
            hmc_samples=5000, burn_in=500,
            print_every=500, step_size=None, leapfrog_steps=None,
            lr_decay_step=2000, lr_decay_gamma=0.5,
            **unused):

        # ── hyper-parameters ────────────────────────────────────────────
        if step_size   is not None:  self.eps = step_size
        if leapfrog_steps is not None: self.L = leapfrog_steps

        self._coloc_pt_num = coloc_pt_num
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

        # ── 1️⃣  MAP optimisation ───────────────────────────────────────
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=lr_decay_step, gamma=lr_decay_gamma
        )
        hist = {k: [] for k in ("Total", "Data", "PDE", "IC", "BC")}
        
        map_bar = trange(1, epochs + 1, desc="MAP", leave=False)
        for ep in map_bar:
            opt.zero_grad()
            logL, parts = self._log_likelihood(X_train, Y_train,
                                            lam_pde, lam_ic, lam_bc, lam_data)
            nlp = -(self._log_prior(flatten_params(self.parameters())) + logL)
            nlp.backward()
            opt.step()
            scheduler.step()

            hist["Total"].append(nlp.item())
            for k, v in parts.items():
                hist[k].append(v.item())

            if ep % print_every == 0 or ep == epochs:
                map_bar.write(
                    f"[MAP] epoch {ep:6d} −logPost={nlp.item():.3e}  "
                    + "  ".join(f"{k}={parts[k]:.2e}" for k in parts)
                )
            map_bar.set_postfix(loss=f"{nlp.item():.2e}")

        # ── 2️⃣  HMC helpers ────────────────────────────────────────────
        θ0 = flatten_params(self.parameters()).detach().to(self.device)
        θ0 = θ0.repeat(self.chains, 1)                       # (B, D)

        def potential_and_grad(theta_mat):
            """
            theta_mat : (B, D)
            returns   : U (B,), grad_U (B, D)
            """
            U_vals, grads = [], []
            for θ in theta_mat:
                # write θ into the network (no autograd link needed)
                unflatten_params(self, θ)

                logL, _ = self._log_likelihood(
                    X_train, Y_train, lam_pde, lam_ic, lam_bc, lam_data
                )
                U = -(self._log_prior(θ) + logL)

                # clear & back-prop to collect param-grads
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                U.backward()
                g = flatten_params(p.grad for p in self.parameters())

                U_vals.append(U.detach())
                grads.append(g.detach())
            return torch.stack(U_vals), torch.stack(grads)

        # ── 3️⃣  HMC sampling ────────────────────────────────────────────
        self._posterior = []                      # keep list format
        θ = θ0.clone()
        accept_cnt = 0

        hmc_bar = trange(1, hmc_samples + 1, desc="HMC", leave=False)
        for it in hmc_bar:
            p0 = torch.randn_like(θ)
            θ_prop, p = θ.clone(), p0.clone()

            # current energy & grad
            U_curr, g = potential_and_grad(θ_prop)

            # —— Leap-frog integrator ——
            p = p - 0.5 * self.eps * g
            for lf in range(self.L):
                θ_prop = θ_prop + self.eps * p
                U_prop, g = potential_and_grad(θ_prop)
                if lf != self.L - 1:
                    p = p - self.eps * g
            p = p - 0.5 * self.eps * g
            p = -p                                         # momentum flip

            # —— Metropolis acceptance ——
            K0    = 0.5 * p0.pow(2).sum(dim=1)
            Kprop = 0.5 * p.pow(2).sum(dim=1)
            delta_H = (U_prop + Kprop) - (U_curr + K0)
            acc_prob = torch.exp(-delta_H).clamp(max=1.0)
            accept   = (torch.rand_like(acc_prob) < acc_prob).view(-1, 1)

            θ = torch.where(accept, θ_prop, θ)
            accept_cnt += accept.float().sum().item()

            # keep samples after burn-in
            if it > burn_in:
                self._posterior.append(θ.clone().cpu())    # list element (B, D)

            # progress bar
            if it % print_every == 0 or it == hmc_samples:
                hmc_bar.write(f"[HMC] iter {it:6d}  acc-rate={accept_cnt/it:.2f}")
            hmc_bar.set_postfix(acc=f"{accept_cnt/it:.2f}")

        print(f"Finished HMC: avg acceptance "
            f"{accept_cnt / (hmc_samples * self.chains):.3f}")
        print(f"Kept {len(self._posterior)} posterior samples")

        # restore the final θ to the network
        unflatten_params(self, θ[0])
        return hist

    @torch.no_grad()
    def predict(self, alpha, X_test,
                X_train=None, Y_train=None,
                X_cal=None,  Y_cal=None,
                heuristic_u=None, k=None,
                n_samples=5_000, if_return_mean=False):
        if not self._posterior:
            y = self(X_test)
            return (y, y)

        saved_params = flatten_params(self.parameters()).detach().clone()
        expected_numel = saved_params.numel()
        preds = []

        # Handle posterior storage format: list of [D] or a single [B,D] tensor
        if isinstance(self._posterior, list):
            posterior = torch.stack(self._posterior)
        else:
            posterior = self._posterior  # assume tensor already

        sample_idx = torch.randint(0, posterior.shape[0], (n_samples,), device=self.device)
        for idx in sample_idx:
            theta = posterior[idx].flatten().to(self.device)

            if theta.numel() != expected_numel:
                raise RuntimeError(f"Posterior θ shape mismatch: expected {expected_numel}, got {theta.numel()}")
            unflatten_params(self, theta)
            preds.append(self(X_test))

        unflatten_params(self, saved_params)

        preds = torch.stack(preds)  # (S, N, 1)
        μ, σ = preds.mean(0), preds.std(0)
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha / 2)).abs()

        if if_return_mean:
            return (μ - z * σ, μ + z * σ), μ
        else:
            return (μ - z * σ, μ + z * σ)



    # ------------------------------------------------------------------
    # evaluation: test MSE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def data_loss(self, X_test, Y_test):
        """Compute the data loss (MSE) on the given dataset."""
        X_test = X_test.to(self.device)
        Y_test = Y_test.to(self.device)
        preds  = self(X_test)
        loss   = torch.nn.functional.mse_loss(preds, Y_test, reduction="mean")
        return loss.item()
