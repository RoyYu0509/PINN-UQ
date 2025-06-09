from utils_model_pinn import *

class DropoutPINN_MLP_1D(PINN_MLP_1D):
    def __init__(self, zeta: float, omega: float, 
                 layers=(1, 64, 64, 64, 1), 
                 activation=torch.tanh,
                 learn_pde_var=False,
                 p_drop=0.1):
        super().__init__(zeta, omega, layers, activation, learn_pde_var)
        
        self.dropout_rate = p_drop

        # Convert function to module
        if activation == torch.tanh:
            act_module = nn.Tanh()
        elif activation == torch.relu:
            act_module = nn.ReLU()
        elif activation == torch.sigmoid:
            act_module = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation. Use torch.tanh, torch.relu, or torch.sigmoid.")

        # Redefine layers with dropout
        self.input_layer = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            act_module,
            nn.Dropout(p_drop)
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                act_module,
                nn.Dropout(p_drop)
            ))

        self.output_layer = nn.Linear(layers[-2], layers[-1])
    
    # Redefine the forward() method
    def forward(self, x, return_hidden=False):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        hidden = x
        out = self.output_layer(hidden)
        return (out, hidden) if return_hidden else out

    # -------------- utility ----------
    def _set_dropout(self, active: bool):
        """
        If active=True  ➜ model.train()  (dropout ON)
        If active=False ➜ model.eval()   (dropout OFF)
        """
        if active:
            super().train()   # keep everything else identical
        else:
            super().eval()

    # ------------- TRAINING ----------
    def fit_pinn_oscillator(self, *args, drop_activate=False ,**kwargs):
        # 1 deactivate the dropout during model training
        self._set_dropout(active=drop_activate)

        # 2 call the parent implementation (unchanged)
        return super().fit_pinn_oscillator(*args, **kwargs)

    # ------------- INFERENCE ----------
    @torch.no_grad()
    def predict(self, t, mc_samples=100, return_std=False, drop_activate=False):
        if drop_activate:
            # 1 reactivate dropout *only* for MC-sampling  
            self._set_dropout(active=drop_activate)
            
            # Conduct MCMC sampling using the model with drop out activate
            preds = []
            for _ in range(mc_samples):
                y = super(DropoutPINN_MLP_1D, self).forward(t.float())
                if isinstance(y, tuple):
                    y = y[0]
                preds.append(y.cpu().numpy())

            # 2 switch back to eval to avoid surprises later
            self._set_dropout(active=False)

            preds = np.stack(preds, axis=0)
            mean = preds.mean(axis=0)
            std  = preds.std(axis=0)
        else:                                 # deterministic
            self._set_dropout(False)
            y = super().forward(t)
            y = y[0] if isinstance(y, tuple) else y
            mean = y
            std  = torch.zeros_like(mean)
        return (mean, std) if return_std else mean