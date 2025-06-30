import os
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt


import os
import matplotlib.pyplot as plt

def save_plot(plot_fn, save_dir="plots", prefix="plot", params=None, loss=None, fmt="png"):
    """
    Returns a wrapped plotting function that saves the plot with hyperparameter info.

    Args:
        plot_fn: the original plotting function.
        save_dir: folder to save plots.
        prefix: prefix in filename.
        params: dictionary of hyperparameters.
        loss: optional loss value for filename.
        fmt: file format.

    Returns:
        A function that wraps the original plot function and saves the figure.
    """
    def wrapped_plot_fn(*args, **kwargs):
        os.makedirs(save_dir, exist_ok=True)

        # Format filename
        param_str = ",".join(f"{k}_{str(v).replace('.', '_')}" for k, v in (params or {}).items())
        loss_str = f";loss_{loss:.3f}" if loss is not None else ""
        filename = " ".join(filter(None, [prefix, param_str])) + f"{loss_str}" + f".{fmt}"
        filepath = os.path.join(save_dir, filename)

        # Plot and save
        plt.figure()
        plot_fn(*args, **kwargs)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        print(f"[✅] Saved plot to: {filepath}")
        
    return wrapped_plot_fn


def hyperparameter_tuning(
        plotting_func,
        plot_title,
        uqmodel, alpha, 
        X_test, Y_test, 
        fit_args: dict, fit_kwargs_grid: dict, uqmd_pred_kwargs: dict, cp_pred_kwargs: dict, 
        save_dir="uqmodel"
    ):
    """
    Performs grid search over hyperparameters, trains model, and saves prediction plot for each config.

    Args:
        uqmodel: An instance of your model class (must have fit(...) and predict(...) methods).
        alpha: significance level
        X_test: the input test grid
        fit_args: Fixed arguments to fit (dict, e.g., {"X_train": ..., "Y_train": ..., "epochs": 1000}).
        fit_kwargs_grid: Dict of hyperparameter name → list of values to try (e.g., {"lr": [1e-3, 1e-4]}).
        uqmd_pred_kwargs: the kwargs for the baseline uq model
        save_dir: Folder to store the saved plots.

    Returns:
        best_params: The hyperparameter combination with the lowest loss.
    """

    best_loss = float("inf")
    best_params = None

    # Generate all combinations of hyperparameter values
    keys, values = zip(*fit_kwargs_grid.items())
    combinations = list(itertools.product(*values))

    for combo in combinations:
        hyperparams = dict(zip(keys, combo))
        print(f"\n[🔍] Trying: {hyperparams}")

        # Baseline Model
        print(f"\n[👟] Training: Baseline {save_dir}")
        baseline_loss_dict = uqmodel.fit(**fit_args, **hyperparams)
        baseline_data_loss = baseline_loss_dict["Data"]
        
        print(f"\n[👟] Predicting: Baseline {save_dir}")
        # Baseline Model Prediction
        cp_uncal_predset = do_pinn.predict(
            alpha=alpha, X_test=X_test,
            **uqmd_pred_kwargs
        )

        # CP+ Model
        cp_model = CP(uqmodel)

        # CP+ Model Prediction
        print(f"\n[🔮] Predicting: CP + {save_dir}")
        cp_cal_predset = cp_model.predict(
            alpha=alpha, X_test=X_test,
            **cp_pred_kwargs
        )

        # Update best
        if baseline_data_loss < best_loss:
            best_loss = baseline_data_loss
            best_params = hyperparams

        # Save the plot using a wrapper
        save_plot(
            plotting_func,
            save_dir=save_dir, prefix=save_dir,
            params=hyperparams, 
            loss=best_loss
        )(X_test, cp_uncal_predset, cp_cal_predset, true_solution, title=plot_title)

    print(f"\n[🏆] Best Hyperparameters: {best_params} with Loss: {best_loss:.4f}")
    return best_params
