import os
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import gc
import torch
import copy


def clear_memory(*vars_to_delete):
    """Delete specified variables and run garbage collector (CPU-only version)."""
    for var in vars_to_delete:
        try:
            del var
        except:
            pass
    gc.collect()


def save_plot(plot_fn, save_dir="plots", prefix="plot", params=None, loss=None, fmt="pdf"):
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
        loss_str = f"loss_{loss:.3e}" if loss is not None else ""
        filename = "Model: ".join(filter(None, [prefix, param_str])) + f"; {loss_str}" + f".{fmt}"
        filepath = os.path.join(save_dir, filename)

        # Plot and save
        plt.figure()
        plot_fn(*args, **kwargs)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        # print(f"[✅] Saved plot to: {filepath}")
        
    return wrapped_plot_fn


import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from utils_tools.utils_result_viz import plot_2D_comparison_with_coverage
from utils_uqmd.utils_uq_cp import CP
from utils_tools.utils_result_metrics import cp_test_uncertainties, vi_test_uncertainties, do_test_uncertainties


def hyperparameter_tuning(
        plot_title,
        # Model Fitting & Predicting
        uqmodel, alpha, 
        X_test, Y_test,
        fit_args: dict, fit_kwargs_grid: dict,
        baseline_pred_kwargs: dict, cp_pred_kwargs: dict,
        true_solution,
        # Coverage Test
        baseline_testing_args: dict, cp_testing_args: dict,
        baseline_test_uncertainties,
        # Plotting function
        plotting_func = plot_2D_comparison_with_coverage,
        save_dir="uqmodel",
        X_vis=None, Y_vis=None,
        # Needed for model selection
        X_validation=None, Y_validation=None,
    ):
    """
    Performs grid search over hyperparameters, trains model, and saves prediction plot for each config.

    Args:
        uqmodel: An instance of your model class (must have fit(...) and predict(...) methods).
        alpha: significance level
        X_test: the input test grid
        fit_args: Fixed arguments to fit (dict, e.g., {"X_train": ..., "Y_train": ..., "epochs": 1000}).
        fit_kwargs_grid: Dict of hyperparameter name → list of values to try (e.g., {"lr": [1e-3, 1e-4]}).
        baseline_pred_kwargs: the kwargs for the baseline uq model
        save_dir: Folder to store the saved plots.

    Returns:
        best_params: The hyperparameter combination with the lowest loss.
    """
    initial_state = copy.deepcopy(uqmodel.state_dict())
    best_loss = float("inf")
    best_params = None

    # Generate all combinations of hyperparameter values
    keys, values = zip(*fit_kwargs_grid.items())
    combinations = list(itertools.product(*values))

    for combo in combinations:
        hyperparams = dict(zip(keys, combo))
        print(f"\n[🔎] Trying: {hyperparams}")
        uqmodel.load_state_dict(initial_state, strict=True)
        uqmodel._posterior = []          # forget any earlier samples

        # Baseline Model
        print(f"\n[🟠] Training...")
        baseline_loss_dict = uqmodel.fit(**fit_args, **hyperparams)

        # Compute the baseline model's data loss
        if (X_validation is None) or (Y_validation is None):
            raise TypeError("Missing validation data `X_validation` or `Y_validation")
        baseline_data_loss = uqmodel.data_loss(X_validation, Y_validation)

        if baseline_data_loss < best_loss:
            best_loss = baseline_data_loss
            best_params = hyperparams

        print(f"\n[🟠] Base Model Inferencing...")
        # Baseline Model Prediction
        cp_uncal_predset = uqmodel.predict(
            alpha=alpha, X_test=X_test,
            **baseline_pred_kwargs
        )
        print(f"\n[🟠] CP Model Inferencing...")
        # CP+ Model
        cp_model = CP(uqmodel)
        # CP+ Model Prediction

        cp_cal_predset = cp_model.predict(
            alpha=alpha, X_test=X_test,
            **cp_pred_kwargs
        )
    
        # Compute the metrics and coverage plots
        print(f"\n[🟠] Computing Coverage...")
        df_uncal = baseline_test_uncertainties(uqmodel=uqmodel, **baseline_testing_args)
        df_cal = cp_test_uncertainties(cp_model, **cp_testing_args)
        
        print(f"\n[✅] Data Loss = {baseline_data_loss:.3e}")

        main_title = f"Loss: {baseline_data_loss:.3e}, {hyperparams}"

        # Save the plot using a wrapper
        save_plot(
            plotting_func,
            save_dir=save_dir, prefix=save_dir,
            params=hyperparams, 
            loss=baseline_data_loss
        )(X_test, cp_uncal_predset, cp_cal_predset, true_solution, df_uncal, df_cal,
          title=plot_title, main_title=main_title, X_vis=X_vis, Y_vis=Y_vis)
        
        clear_memory(
            cp_model,
            cp_uncal_predset, cp_cal_predset,
            df_uncal, df_cal,
            plotting_func
        )

    print(f"\n[🏆] Best Hyperparameters: {best_params} with Loss: {best_loss:.4f}")
    return best_params




def hyperparameter_tuning_higher_dimensional(
        plot_title,
        # Model Fitting & Predicting
        uqmodel, alpha, 
        X_test, Y_test,
        fit_args: dict, fit_kwargs_grid: dict,
        baseline_pred_kwargs: dict, cp_pred_kwargs: dict,
        true_solution,
        # Coverage Test
        baseline_testing_args: dict, cp_testing_args: dict,
        baseline_test_uncertainties,
        # Plotting function
        plotting_func = plot_2D_comparison_with_coverage,
        save_dir="uqmodel",
        X_vis=None, Y_vis=None,
        # Needed for model selection
        X_validation=None, Y_validation=None,
    ):
    """
    Performs grid search over hyperparameters, trains model, and saves prediction plot for each config.

    Args:
        uqmodel: An instance of your model class (must have fit(...) and predict(...) methods).
        alpha: significance level
        X_test: the input test grid
        fit_args: Fixed arguments to fit (dict, e.g., {"X_train": ..., "Y_train": ..., "epochs": 1000}).
        fit_kwargs_grid: Dict of hyperparameter name → list of values to try (e.g., {"lr": [1e-3, 1e-4]}).
        baseline_pred_kwargs: the kwargs for the baseline uq model
        save_dir: Folder to store the saved plots.

    Returns:
        best_params: The hyperparameter combination with the lowest loss.
    """
    initial_state = copy.deepcopy(uqmodel.state_dict())
    best_loss = float("inf")
    best_params = None

    # Generate all combinations of hyperparameter values
    keys, values = zip(*fit_kwargs_grid.items())
    combinations = list(itertools.product(*values))

    for combo in combinations:
        # uqmodel = copy.deepcopy(uqmodel)
        hyperparams = dict(zip(keys, combo))
        print(f"\n[🔎] Trying: {hyperparams}")
        uqmodel.load_state_dict(initial_state, strict=True)
        uqmodel._posterior = [] 
        
        # Baseline Model
        print(f"\n[🟠] Training...")
        baseline_loss_dict = uqmodel.fit(**fit_args, **hyperparams)

        # Compute the baseline model's data loss
        if (X_validation is None) or (Y_validation is None):
            raise TypeError("Missing validation data `X_validation` or `Y_validation")
        baseline_data_loss = uqmodel.data_loss(X_validation, Y_validation)

        if baseline_data_loss < best_loss:
            best_loss = baseline_data_loss
            best_params = hyperparams

        print(f"\n[🟠] Inferencing...")
        # Baseline Model Prediction
        cp_uncal_predset = uqmodel.predict(
            alpha=alpha, X_test=X_test,
            **baseline_pred_kwargs
        )
        # CP+ Model
        cp_model = CP(uqmodel)
        # CP+ Model Prediction
        cp_cal_predset = cp_model.predict(
            alpha=alpha, X_test=X_test,
            **cp_pred_kwargs
        )
    
        # Compute the metrics and coverage plots
        print(f"\n[🟠] Computing Coverage...")
        df_uncal = baseline_test_uncertainties(**baseline_testing_args)
        df_cal = cp_test_uncertainties(cp_model, **cp_testing_args)
        
        print(f"\n[✅] Data Loss = {baseline_data_loss:.3e}")

        # Return the metrics of the model at 95% confidence level
        uncal_95rslt_row = df_uncal.loc[0.9<=df_uncal['alpha'] & df_uncal['alpha']<=0.95]
        cal_95rslt_row = df_cal.loc[0.9<=df_cal['alpha'] & df_cal['alpha']<=0.95]

        main_title = f"Loss: {baseline_data_loss:.3e}, {hyperparams}"
        # TODO: plot the 2 rslt row
        # Save the plot using a wrapper
        save_plot(
            plotting_func,
            save_dir=save_dir, prefix=save_dir,
            params=hyperparams, 
            loss=baseline_data_loss
        )(uncal_95rslt_row, cal_95rslt_row)

        clear_memory(
            uqmodel, cp_model,
            cp_uncal_predset, cp_cal_predset,
            df_uncal, df_cal,
            plotting_func
        )
        

    print(f"\n[🏆] Best Hyperparameters: {best_params} with Loss: {best_loss:.4f}")
    return best_params