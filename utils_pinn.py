#Importing the necessary 
import os
import numpy as np 
import math
from tqdm import tqdm 
from timeit import default_timer
import matplotlib as mpl 
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import random

import torch, math, matplotlib.pyplot as plt
from torch import nn
# from pyDOE import lhs
# from utils import *

import sklearn
from sklearn.neighbors import NearestNeighbors

from itertools import chain, combinations
import torch.nn as nn
import math

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from typing import Sequence, Tuple, List, Union, Callable
####################################################


class MLP(nn.Module):
    def __init__(self, zeta: float, omega: float, 
                 layers=(1, 64, 64, 64, 1), activation=torch.tanh, learn_pde_var=False):
        super().__init__()
        self.in_features = layers[0]
        self.out_features = layers[-1]
        self.num_layers = len(layers)
        self.act_func = activation

        self.input_layer = nn.Linear(layers[0], layers[1])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(1, self.num_layers - 2)
        ])
        self.output_layer = nn.Linear(layers[-2], layers[-1])

        if learn_pde_var:
            # Learnable physical parameters
            self.zeta = nn.Parameter(torch.tensor(zeta))   # initial guess
            self.omega = nn.Parameter(torch.tensor(omega))  # initial guess
        else:
            self.zeta = torch.tensor(zeta)  # initial guess
            self.omega = torch.tensor(omega)  # initial guess


    def forward(self, x, return_hidden=False):
        x = self.act_func(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.act_func(layer(x))
        hidden = x
        out = self.output_layer(hidden)
        return (out, hidden) if return_hidden else out
    

class PINN_MLP_1D(MLP):
    """Learn a PINN model for different 1d PDE"""

    def __init__(self, zeta: float, omega: float, layers, activation=torch.tanh, learn_pde_var=False):
        super().__init__(zeta, omega, layers, activation, learn_pde_var)
    
    def fit_pinn_oscillator(self, 
        t_c, t_d, x_d,
        t0, x0, v0,
        λ_pde = 10.0, λ_ic = 5.0, λ_data = 5.0,
        epochs = 20_000,
        lr = 3e-3,
        print_every = 500,
        scheduler_cls = StepLR,
        scheduler_kwargs = {'step_size': 5000, 'gamma': 0.5},
        warm_up_steps = 5000):

        self.to(device)  # move model to device

        # Ensure tensors are float32 and on correct device
        t_c = t_c.to(dtype=torch.float32, device=device).requires_grad_()
        t_d = t_d.to(dtype=torch.float32, device=device)
        x_d = x_d.to(dtype=torch.float32, device=device)
        t0  = t0.to(dtype=torch.float32, device=device).requires_grad_()
        x0  = x0.to(dtype=torch.float32, device=device)
        v0  = v0.to(dtype=torch.float32, device=device)

        # Optimizer
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # Scheduler
        scheduler = scheduler_cls(opt, **scheduler_kwargs) if scheduler_cls else None

        # Training History
        tot_loss_his = []
        pde_loss_his = []
        ic_loss_his = []
        data_loss_his = []

        for ep in range(1, epochs + 1):
            zeta = self.zeta
            omega = self.omega

            opt.zero_grad()

            # Warm-up Option: Loss Function with Warm-up Phase
            if ep <= warm_up_steps:
                # Data loss
                x_pred = self(t_d)
                loss_data = ((x_pred - x_d) ** 2).mean()
                loss = loss_data

            else:
                # Data loss
                x_pred = self(t_d)
                loss_data = ((x_pred - x_d) ** 2).mean()

                # PDE residual
                x_colloc = self(t_c)
                dx_dt = torch.autograd.grad(x_colloc, t_c, torch.ones_like(x_colloc), create_graph=True)[0]
                d2x_dt2 = torch.autograd.grad(dx_dt, t_c, torch.ones_like(dx_dt), create_graph=True)[0]
                residual = d2x_dt2 + 2 * zeta * omega * dx_dt + (omega**2) * x_colloc
                loss_pde = (residual ** 2).mean()

                # Initial conditions
                x0_pred = self(t0)
                dx0_pred = torch.autograd.grad(x0_pred, t0, torch.ones_like(x0_pred), create_graph=True)[0]
                loss_ic = (x0_pred - x0) ** 2 + (dx0_pred - v0) ** 2

                # Total loss
                loss = λ_pde * loss_pde + λ_data * loss_data + λ_ic * loss_ic

            loss.backward()
            opt.step()

            if ep <= 40000:  # Stop decreasing the learning rate
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(loss.item())
                    else:
                        scheduler.step()

            if (ep % print_every == 0 or ep == 1) and ep>warm_up_steps:  # Only start reporting after the warm-up Phase
                print(f"ep {ep:5d} | L={loss.item():.2e}  "
                    f"data={loss_data.item():.2e}  pde={loss_pde.item():.2e}  "
                    f"ic={loss_ic.item():.2e} | lr={opt.param_groups[0]['lr']:.2e} |"
                    f"zeta={zeta.item():.3e}, omega={omega.item():.3e}")
                
                tot_loss_his.append(loss.item())
                pde_loss_his.append(loss_pde.item())
                ic_loss_his.append(loss_ic.item())
                data_loss_his.append(loss_data.item())

            elif (ep % print_every == 0 or ep == 1): 
                print(f"ep {ep:5d} | L={loss.item():.2e}  "
                    f"data={loss_data.item():.2e}")
                
            return data_loss_his, ic_loss_his, pde_loss_his