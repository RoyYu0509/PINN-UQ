import math, torch, torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from abc import ABC, abstractmethod                     # ← enforce required methods

class BaseUQPINN(nn.Module, ABC):
    """
    Abstract parent for all uncertainty-quantified PINNs (VI, MC-Dropout, Deep-Ensemble,
    HMC, CQR, etc.).  Every subclass must implement the methods marked *abstractmethod*.
    """

    # ────────────────────────────────────────────────────────────────
    # 1. INITIALISATION
    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        pde_class,                      # ⇨ object with .residual(…), .boundary_loss(…), .ic_loss(…)
        input_dim:   int,
        hidden_dims: list[int] | int,
        output_dim:  int,
        act_func:    nn.Module = nn.Tanh(),
        **uq_kwargs,                    # ⇨ anything specific to the chosen UQ scheme
    ):
        super().__init__()
        self.pde = pde_class

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [self._make_linear(prev, h, **uq_kwargs), act_func]
            prev = h
        layers += [self._make_linear(prev, output_dim, **uq_kwargs)]
        self.layers = nn.ModuleList(layers)

        # Optional learnable data-noise handled uniformly:
        init_noise = uq_kwargs.get("init_data_noise", 1.0)
        self.learn_data_noise = uq_kwargs.get("learn_data_noise", False)
        if self.learn_data_noise:
            self.log_noise = nn.Parameter(torch.tensor(math.log(init_noise)))
        else:
            self.register_buffer("log_noise", torch.tensor(math.log(init_noise)))

    # ────────────────────────────────────────────────────────────────
    # 2. BUILDING-BLOCK FACTORY
    #    ↳ Each UQ flavour overrides this if it needs BayesianLinear,
    #      dropout, ensembles, etc.
    # ────────────────────────────────────────────────────────────────
    @abstractmethod
    def _make_linear(self, in_f, out_f, **uq_kwargs) -> nn.Module:
        """Return one fully-connected layer *including* whatever UQ wrapper it needs."""
        raise NotImplementedError

    # ────────────────────────────────────────────────────────────────
    # 3. CORE MODEL OPS
    # ────────────────────────────────────────────────────────────────
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # UQ-specific bookkeeping needed by training loop
    @abstractmethod
    def kl_divergence(self) -> torch.Tensor:
        """Return KL(q‖p) or an equivalent regulariser (0 for non-Bayesian methods)."""
        raise NotImplementedError

    # Generic Gaussian log-likelihood
    def nll_gaussian(self, y_pred, y_true, noise_std):
        return 0.5 * ((y_pred - y_true) ** 2).mean() / (noise_std ** 2)

    # ────────────────────────────────────────────────────────────────
    # 4. TRAINING LOOP
    #    ↳ identical signature across subclasses so caller code
    #      never has to branch.
    # ────────────────────────────────────────────────────────────────
    def fit(
        self,
        coloc_pt_num: int,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        *,
        λ_pde=1.0, λ_ic=1.0, λ_bc=1.0, λ_elbo=1.0,
        epochs=20_000, lr=3e-3,
        scheduler_cls=StepLR,
        scheduler_kwargs={"step_size": 5_000, "gamma": 0.5},
        stop_schedule=40_000,
        **extra_fit_kw,                 # ← e.g. dropout_rate, ensemble_size…
    ):
        opt        = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler  = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None
        histories  = {"ELBO": [], "IC": [], "BC": [], "PDE": []}
        self.train()

        for ep in range(epochs):
            opt.zero_grad()

            # 4.1 Physics terms (if PDE supplies them)
            loss_pde = self._maybe(self.pde, "residual", self, coloc_pt_num)
            loss_bc  = self._maybe(self.pde, "boundary_loss", self)
            loss_ic  = self._maybe(self.pde, "ic_loss",      self)

            # 4.2 Data term
            y_pred    = self.forward(X_train)
            noise_std = torch.exp(self.log_noise)
            loss_data = self.nll_gaussian(y_pred, Y_train, noise_std)

            # 4.3 Regularisation (KL or equivalent)
            kl = self.kl_divergence() / (coloc_pt_num + X_train.shape[0])

            # 4.4 Full objective
            loss = (λ_pde * loss_pde + λ_bc * loss_bc + λ_ic * loss_ic
                    + λ_elbo * (loss_data + kl))
            loss.backward()
            opt.step()

            # Book-keeping
            histories["ELBO"].append((loss_data + kl).item())
            histories["IC"].append(loss_ic.item())
            histories["BC"].append(loss_bc.item())
            histories["PDE"].append(loss_pde.item())

            # LR schedule
            if ep <= stop_schedule and scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

        return histories

    # ────────────────────────────────────────────────────────────────
    # 5. PREDICTION INTERFACE
    #    ↳ sample-based, so subclasses just override _single_forward
    # ────────────────────────────────────────────────────────────────
    def predict(self, x_test, *, n_samples=100, z_score=1.96, **sample_kw):
        self.eval()
        preds = torch.stack([self._single_forward(x_test, **sample_kw)
                             for _ in range(n_samples)])

        mean, std = preds.mean(0), preds.std(0)
        return [mean - z_score * std, mean + z_score * std]

    @abstractmethod
    def _single_forward(self, x, **sample_kw):
        """
        Return ONE forward pass with whatever stochasticity is
        required (sampling weights, dropout, bootstrap, etc.).
        """
        raise NotImplementedError

    # ────────────────────────────────────────────────────────────────
    # 6. HELPER
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _maybe(obj, attr, *args):
        return getattr(obj, attr)(*args) if hasattr(obj, attr) else 0.0
