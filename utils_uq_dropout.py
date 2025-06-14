import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# ───────────────────────────────────────────────────────────────────────────────
#  1.  Feed-forward network that inserts Drop-out after every deterministic layer
# ───────────────────────────────────────────────────────────────────────────────

class DeterministicDropoutNN(BasePINNModel):
    """
    Fully–connected network identical to DeterministicFeedForwardNN,
    but each hidden block is Linear ▸ Dropout ▸ Activation.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float = 0.1,
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

class DropoutPINN(PINN):
    """
    Physics-Informed Neural Network with Monte-Carlo Drop-out for UQ.

    Usage
    -----
    >>> model = DropoutPINN(pde, 1, [64,64,64], 1, p_drop=0.05)
    >>> model.fit_pinn(X_train, Y_train, coloc_pt_num, epochs=30_000)
    >>> mean, std, (lower, upper) = model.predict_uq(x_test, n_samples=200, alpha=0.05)
    """
    def __init__(self,
                 pde_class,
                 input_dim: int,
                 hidden_dims,
                 output_dim: int,
                 p_drop: float = 0.1,
                 activation = torch.tanh):
        # Build backbone with dropout layers
        self.backbone = DeterministicDropoutNN(input_dim, hidden_dims,
                                               output_dim, p_drop, activation)
        # Register backbone parameters inside BasePINNModel
        super(PINN, self).__init__()       # skip PINN.__init__, call one level up
        self.__dict__.update(self.backbone.__dict__)  # merge state (quick hack)
        self.pde = pde_class               # physics callbacks

    # -------------------------------------------------------------------------
    # Uncertainty-aware prediction
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def predict_uq(self,
                   x: torch.Tensor,
                   n_samples: int = 100,
                   alpha: float = 0.05,
                   keep_dropout: bool = True,
                   return_samples: bool = False):
        """
        Parameters
        ----------
        x : (N, d_in) tensor on same device / dtype as model
        n_samples : number of MC forward passes
        alpha : 1 – confidence level; alpha=0.05 → 95 % interval
        keep_dropout : if True, forces dropout active during inference
        return_samples : if True, also returns the raw (n_samples, N, out) tensor

        Returns
        -------
        mean  : (N, out)
        std   : (N, out)
        bounds: (lower, upper) each shape (N, out)
        """
        if keep_dropout:
            self.enable_mc_dropout()

        preds = []
        for _ in range(n_samples):
            preds.append(self.forward(x))
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

        if return_samples:
            return mean, std, (lower, upper), preds
        return mean, std, (lower, upper)

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