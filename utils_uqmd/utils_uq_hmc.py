###############################################################################
# HMCBPINN – v2
#     * Works with PDE helpers that expose
#         • residual_losses(model, …)            OR
#         • residual(model, coloc_pt_num)  and  boundary_loss(model, …)
#     * Interface still matches your Dropout-NN and VI-NN for CP utilities
# 
# HMC Best Explaination: https://www.youtube.com/watch?v=FYliDjeYuXg
#
# Intuition:  θ is a position of a particle in space & invent a fake "momentum" 
#             variable p of the same shape
#
# Leap frog: Simulate the progression of θ and p, with dt being replaced by a ε
#            后面那两个乘上的东西是他们的derivative
# 
###############################################################################
import copy, torch, math
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

class HMCBPINN(nn.Module):
    # ───────────────────────── constructor ────────────────────────────
    def __init__(self,
                 pde_class,
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int,
                 act_func      = nn.Tanh(),
                 prior_std     : float = 1.0,
                 step_size     : float = 1e-3,
                 leapfrog_steps: int   = 5,
                 device=None):
        super().__init__()

        self.pde            = pde_class
        self.prior_std      = prior_std
        self.step_size      = step_size
        self.leapfrog_steps = leapfrog_steps
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")

        layers = [nn.Linear(input_dim, hidden_dims[0]), act_func]
        for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers += [nn.Linear(h_in, h_out), act_func]
        layers += [nn.Linear(hidden_dims[-1], output_dim)]
        self.net = nn.Sequential(*layers).to(self.device)

        self._posterior : list[dict[str, torch.Tensor]] = []
        self._coloc_pt_num = None      # set during fit()

    # ────────────────────────── forward ───────────────────────────────
    def forward(self, x, *, return_hidden=False):
        h = x.to(self.device)
        hidden = []
        for layer in self.net:
            h = layer(h)
            if return_hidden and isinstance(layer, nn.Linear):
                hidden.append(h)
        return (h, torch.cat(hidden, -1)) if return_hidden else h

    # ───────────────── probability components ────────────────────────
    def _log_prior(self):
        # Assume theta ~ N(0, σ2)
        σ2 = self.prior_std ** 2
        # Only lefted with the non-constant term, and it acts like regualrization term in the final loss
        return sum((-(p**2).sum() / (2*σ2)) for p in self.parameters())

    def _pde_losses(self, coloc_pt_num):
        """
        Returns tuple (pde_loss, ic_loss, bc_loss) in **any** of these cases:
          1. PDE helper defines .residual_losses(model, …)
          2. PDE helper defines .residual(model, coloc_pt_num) (+ boundary_loss)
        """
        # Case ➊: unified helper exists
        if hasattr(self.pde, "residual_losses"):
            return self.pde.residual_losses(self)     # may accept defaults

        # Case ➋: use residual() + boundary_loss()
        device = next(self.parameters()).device
        pde_loss = torch.tensor(0.0, device=device)
        bc_loss  = torch.tensor(0.0, device=device)
        ic_loss  = torch.tensor(0.0, device=device)   # Poisson has no IC term

        if hasattr(self.pde, "residual") and coloc_pt_num is not None:
            pde_loss = self.pde.residual(self, coloc_pt_num)
        if hasattr(self.pde, "boundary_loss"):
            bc_loss  = self.pde.boundary_loss(self)

        return pde_loss, ic_loss, bc_loss

    def _log_likelihood(self, X, Y, λ_pde, λ_ic, λ_bc, λ_data):
        # data term
        data_loss = torch.nn.functional.mse_loss(self(X), Y)

        # physics & boundary
        pde_loss, ic_loss, bc_loss = self._pde_losses(self._coloc_pt_num)

        logL = -(λ_pde*pde_loss + λ_ic*ic_loss + λ_bc*bc_loss + λ_data*data_loss)
        parts = {"PDE": pde_loss, "IC": ic_loss, "BC": bc_loss, "Data": data_loss}
        return logL, parts

    # ───────────── parameter vector helpers (unchanged) ──────────────
    def pack(self):   return torch.cat([p.detach().flatten() for p in self.parameters()])
    def unpack(self, vec):
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vec[offset:offset+n].view_as(p))
            offset += n

    # ───────────────────────── training ───────────────────────────────
    def fit(self,
            coloc_pt_num,
            X_train, Y_train,
            λ_pde=3.0, λ_ic=10.0, λ_bc=10.0, λ_data=5.0,
            epochs=20_000, lr=3e-3,
            hmc_samples=3_000, burn_in=500,
            step_size=None, leapfrog_steps=None,
            print_every=2_000,
            **unused):
        self._coloc_pt_num = coloc_pt_num
        if step_size is not None:      self.step_size = step_size
        if leapfrog_steps is not None: self.leapfrog_steps = leapfrog_steps

        hist = {"Total": [], "Data": [], "PDE": [], "IC": [], "BC": []}

        # 1️⃣  MAP phase --------------------------------------------------------
        opt   = torch.optim.Adam(self.parameters(), lr=lr)
        sched = StepLR(opt, step_size=5_000, gamma=0.5)

        for ep in range(epochs):
            opt.zero_grad()
            logL, parts = self._log_likelihood(X_train, Y_train,
                                               λ_pde, λ_ic, λ_bc, λ_data)
            nlp = -(self._log_prior() + logL)
            # Descent on the negative log probability
            nlp.backward(); opt.step(); sched.step()

            hist["Total"].append(nlp.item())
            for k in hist.keys() - {"Total"}:
                v = parts[k]; hist[k].append(v.item() if torch.is_tensor(v) else 0.)
            
            if (ep+1) % print_every == 0:
                print(f"[MAP]  epoch {ep+1:6d}   −logPost={nlp.item():.3e}")

        # 2️⃣  HMC sampling -----------------------------------------------------
        params = list(self.parameters())
        ε, L   = self.step_size, self.leapfrog_steps
        acc    = 0

        def U():   # potential energy
            return -(self._log_prior() +
                     self._log_likelihood(
                        X_train, Y_train,
                        λ_pde, λ_ic, λ_bc, λ_data
                        )[0] # acess the likelihood value
                    )

        # Start sampling
        for it in range(hmc_samples):
            # Use torch.randn_like(p) to sample from N(0,1) for all the parameters
            mom = [torch.randn_like(p) for p in params]      # momenta

            # Create copies of current parameters values (initial)
            θ0  = [p.detach().clone() for p in params]
            # Create copies of current momentum (initial)
            m0  = [m.clone() for m in mom]

            # Compute the current Kinetic & Potential energy (initial)
            U0 = U(); K0 = sum((m**2).sum() for m in mom) * 0.5


            # p - theta parameters (position); m - momentum (velocity)
            # half-kick: compute (m -= ε ∇U/2 ) in high dimensional space
            self.zero_grad(); U0.backward()
            for p, m in zip(params, mom):
                m.sub_(0.5 * ε * p.grad) 

            # leap-frog
            for l in range(L):
                # Compute: θ = (θ + ε p) in high dimensional space
                for p, m in zip(params, mom): 
                    p.data.add_(ε * m)  
                
                # Clear 上一个 step 的 gradient info 
                self.zero_grad() # Empty the grad info before a new step

                # 重新 evaluate potential energy, 因为现在的 p = [θ....] 不一样了
                Umid = U() # re-evaluate the potential energy
                Umid.backward() # fill the p.grad with the new gradient.

                # Compute (p − ε ∇U) in high dimensional space
                for p, m in zip(params, mom):
                    g = p.grad
                    if l < L-1:
                        m.sub_(ε * g)
                    # Close up by only performing half update
                    else:
                         m.sub_(0.5 * ε * g)
            
            # Compute U and K at the proposed (θ, p)
            U1 = U(); K1 = sum((m**2).sum() for m in mom) * 0.5
            # Acceptance prob
            a  = torch.exp((U0 + K0) - (U1 + K1)).clamp(max=1)

            # If accept
            if torch.rand([]) < a: 
                acc += 1
            # If reject, fall back to original (θ, p)
            else: 
                for p, θ in zip(params, θ0): 
                    p.data.copy_(θ)
            
            # 如果过了 burn_in period, store 这次 sample 出来的结果
            if it >= burn_in:
                self._posterior.append(copy.deepcopy(self.state_dict()))

            if (it+1) % print_every == 0:
                print(f"[HMC] iter {it+1:6d}   acc-rate={acc/(it+1):.2f}")

        print(f"HMC finished – kept {len(self._posterior)} posterior samples.")
        return hist

    # ───────────────────────── inference ─────────────────────────────
    @torch.inference_mode()
    def predict(self, alpha, X_test,
                X_train=None, Y_train=None,
                X_cal=None,  Y_cal=None,
                heuristic_u=None, k=None,
                n_samples=1_000):
        if not self._posterior:
            y = self(X_test); return (y, y)
    
        sample_idx = torch.randint(0, len(self._posterior),
                            (n_samples,), device=self.device)
        saved = copy.deepcopy(self.state_dict())
        preds = []
        for idx in sample_idx:
            # load the sample parameters to the model
            self.load_state_dict(self._posterior[int(idx)])
            # use the current parameters model to predict
            preds.append(self(X_test))

        # Reload original parameters set (不重要, 只是确保我们不该掉任何随机性)
        self.load_state_dict(saved)
        preds = torch.stack(preds)                 # (S,N,1)
        μ, σ  = preds.mean(0), preds.std(0)
        z     = torch.distributions.Normal(0,1).icdf(torch.tensor(alpha/2)).abs()
        return μ - z*σ, μ + z*σ
