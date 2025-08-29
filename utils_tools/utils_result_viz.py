import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_truth_and_samples_1D(
    x0: float,
    x1: float,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    true_solution,
    x_colloc,
    n_grid: int = 500,
    title: str = "True solution vs. training data",
    show: bool = False,
):
    """
    Parameters
    ----------
    true_solution : callable
        Analytic solution u*(t) that accepts a NumPy array and returns a NumPy array.
    x0, x1 : float
        Domain limits along the time axis.
    X_train, Y_train : torch.Tensor
        Training inputs and targets (any device; will be moved to CPU for plotting).
    n_grid : int, default 500
        Number of points for the smooth reference curve.
    title : str
        Figure title.
    show : bool, default True
        If True, immediately shows the plot; otherwise returns the figure for further styling.
    """
    # --- prepare dense grid for the analytic curve ------------------------
    t_dense = np.linspace(x0, x1, n_grid)
    y_dense = true_solution(t_dense)

    # --- convert training tensors to NumPy on CPU -------------------------
    t_train = X_train.detach().cpu().numpy().ravel()
    y_train = Y_train.detach().cpu().numpy().ravel()

    x_colloc_np = x_colloc.detach().cpu().numpy().flatten()

    # --- build the figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_dense, y_dense, lw=2, label="true solution")
    ax.scatter(t_train, y_train, s=25, alpha=0.7, label="training points")
    # ax.scatter(x_colloc_np, [0.0] * len(x_colloc_np), color='red', marker='x', label="Collocation points", alpha=0.6)

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    # plt.show()


"""
fig, ax = plot_truth_and_samples(
    x_true_fn = x_true,      # your analytic function
    t_min     = 0.0,
    t_max     = 5.0,
    X_train   = X_train,
    Y_train   = Y_train,
    title     = "Underdamped oscillator: ground-truth vs noisy data"
)
"""

def plot_predictions_1D(x_grid, pred_set, true_solution, main_title="PDE UQ",
                        X=None, Y=None):
    """
    Plots the true solution, predicted bounds (from pred_set), and training data.

    Args:
        x_grid: 1D tensor of test inputs
        pred_set: list of [lower_bound, upper_bound] tensors
        true_solution: function that returns the true u(x)
        title: plot title
        X, Y: optional training data for scatter plot
    """
    x_grid_np = x_grid.detach().cpu().numpy().flatten()
    lower_np = pred_set[0].detach().cpu().numpy().flatten()
    upper_np = pred_set[1].detach().cpu().numpy().flatten()
    mean_np = (lower_np + upper_np) / 2.0

    y_true = true_solution(x_grid_np)

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid_np, y_true, 'k--', label="True solution", linewidth=2)
    plt.plot(x_grid_np, mean_np, 'b-', label="Predicted mean", linewidth=2)
    if X is not None and Y is not None:
        plt.scatter(X.detach().cpu().numpy().flatten(), Y.detach().cpu().numpy().flatten(),
                    c='red', s=20, label="Training data", alpha=0.6)
    plt.fill_between(x_grid_np, lower_np, upper_np, color='blue', alpha=0.3, label="Confidence interval")

    plt.title(main_title)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_training_history(loss_dict, title="Training Loss History", figsize=(10, 6),
                          plot_after=0, step=1):
    """
    Plots stacked area chart of training loss components over epochs.

    Parameters:
    - loss_dict (dict): Dictionary with {loss_name: list of tensor or float values}
    - title (str): Title of the plot
    - figsize (tuple): Size of the figure
    - plot_after (int): Only plot epochs after this index (default: 0)
    - step (int): Plot every `step` epochs (default: 1)
    """
    # Convert all losses to NumPy arrays
    losses_np = {}
    for name, values in loss_dict.items():
        if len(values) == 0:
            continue
        # Convert to NumPy if values are tensors
        arr = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        losses_np[name] = np.array(arr)

    # Make sure all loss arrays are the same length
    lengths = [len(v) for v in losses_np.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All loss lists must be the same length for stacked plotting.")

    total_epochs = lengths[0]
    indices = np.arange(total_epochs)[plot_after::step]

    loss_names = list(losses_np.keys())
    loss_values = np.stack([losses_np[name][indices] for name in loss_names], axis=0)

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(indices, loss_values, labels=loss_names)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()




import matplotlib.pyplot as plt
import pandas as pd

def plot_expected_vs_empirical(df, alpha_col='alpha', cov_col='coverage', title='Coverage Plot'):
    """
    Plots Expected Coverage (1 - alpha) vs Empirical Coverage from a DataFrame,
    and adds manual anchor points at (0, 0) and (1, 1).
    
    Parameters:
    - df: pd.DataFrame with 'alpha' and 'coverage' columns
    - alpha_col: column name for alpha values
    - cov_col: column name for empirical coverage
    - title: plot title
    """
    # Compute expected and empirical
    expected = 1 - df[alpha_col]
    empirical = df[cov_col]

    # Add anchor points
    expected_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
    empirical_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)

    # Sort by expected for clean line plot
    sorted_idx = expected_full.argsort()
    expected_sorted = expected_full[sorted_idx]
    empirical_sorted = empirical_full[sorted_idx]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(expected_sorted, empirical_sorted, marker='o', label='Empirical Coverage')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y = x)')
    plt.xlabel("Expected Coverage (1 − α)")
    plt.ylabel("Empirical Coverage")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dual_expected_vs_empirical(
    df_uncal,
    df_cal,
    *,
    alpha_col="alpha",
    cov_col="coverage",
    title1="Uncalibrated Model",
    title2="Calibrated Model",
    dev_metric="mae",
    main_title="Coverage: Uncalibrated vs Calibrated",
    figsize=(12, 6),
    constrained=True,           # set False if you prefer tight_layout()
    title_pad=14,               # vertical offset for the figure title
    tight_rect=(0, 0, 1, 0.96), # head-room if constrained=False
):
    """
    Side-by-side coverage plots for uncalibrated and calibrated models.

    The subtitle of each panel shows the deviation from ideal calibration
    (MAE / RMSE / max absolute error).

    Parameters
    ----------
    df_uncal, df_cal : pd.DataFrame
        Must contain columns `alpha_col` and `cov_col`.
    dev_metric : {"mae", "rmse", "max"}
        Metric used for the deviation shown under each panel title.
    main_title : str or None
        Figure-level title.  Set None to suppress it.
    constrained : bool
        Use Matplotlib’s constrained-layout engine (recommended).
        If False, the function falls back to tight_layout with `tight_rect`.
    """

    # ------------------------------------------------------------------
    # helpers
    def _prepare(df):
        exp = 1.0 - df[alpha_col]
        emp = df[cov_col]
        exp_full = pd.concat([pd.Series([0.0]), exp, pd.Series([1.0])],
                             ignore_index=True)
        emp_full = pd.concat([pd.Series([0.0]), emp, pd.Series([1.0])],
                             ignore_index=True)
        order = exp_full.argsort()
        return exp_full[order].to_numpy(), emp_full[order].to_numpy()

    def _deviation(e, m, how):
        diff = np.abs(m - e)
        if how == "mae":
            return diff.mean()
        if how == "rmse":
            return np.sqrt((diff ** 2).mean())
        if how == "max":
            return diff.max()
        raise ValueError("dev_metric must be 'mae', 'rmse', or 'max'")

    # ------------------------------------------------------------------
    # data + metrics
    exp1, emp1 = _prepare(df_uncal)
    exp2, emp2 = _prepare(df_cal)
    dev1 = _deviation(exp1, emp1, dev_metric)
    dev2 = _deviation(exp2, emp2, dev_metric)

    # ------------------------------------------------------------------
    # figure & axes
    if constrained:
        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # panel 1 – uncalibrated
    axes[0].plot(exp1, emp1, marker="o", label="Empirical")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal  y=x")
    axes[0].set_title(f"{title1}")
    axes[0].set_xlabel("Expected Coverage  (1 − α)")
    axes[0].set_ylabel("Empirical Coverage")
    axes[0].grid(True)
    axes[0].legend()

    # panel 2 – calibrated
    axes[1].plot(exp2, emp2, marker="o", label="Empirical")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Ideal  y=x")
    axes[1].set_title(f"{title2}")
    axes[1].set_xlabel("Expected Coverage  (1 − α)")
    axes[1].set_ylabel("Empirical Coverage")
    axes[1].grid(True)
    axes[1].legend()

    # ------------------------------------------------------------------
    # figure-level title and layout finish
    if main_title:
        fig.suptitle(main_title, y=0.96, fontsize=12)

    if not constrained:            # tidy up only if we skipped the CL engine
        fig.tight_layout(rect=tight_rect)

    return fig, axes





# 2D Visualizer
def plot_predictions_2D(XY_test, pred_set, true_solution, title="2D UQ Result"):
    import matplotlib.pyplot as plt
    import numpy as np

    x = XY_test[:, 0].detach().cpu().numpy()
    y = XY_test[:, 1].detach().cpu().numpy()
    lower_np = pred_set[0].detach().cpu().numpy().flatten()
    upper_np = pred_set[1].detach().cpu().numpy().flatten()
    mean_np = (lower_np + upper_np) / 2.0

    true_np = true_solution(XY_test).detach().cpu().numpy().flatten()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title_sub in zip(
        axs,
        [true_np, mean_np, upper_np - lower_np],
        ["True u(x,y)", "Predicted Mean", "Predictive Interval Width"]
    ):
        sc = ax.tricontourf(x, y, data, levels=100)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"{title}: {title_sub}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from scipy.ndimage import zoom


def plot_2D_comparison_with_coverage(
    XY_test,
    pred_set_uncal,
    pred_set_cal,
    true_solution,
    df_uncal,
    df_cal,
    X_pts=None,
    X_vis=None, Y_vis=None,
    title="2D UQ Result",
    vlim_true=None,
    vlim_pred_mean=None,
    vlim_pred_width=None,
    show_pts=False,
    alpha_col='alpha',
    cov_col='coverage',
    metric: str = "mae",
    main_title=None,
    grid_size=20
):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import pandas as pd

    if show_pts:
        if X_vis is not None and Y_vis is not None:
            if isinstance(X_vis, torch.Tensor):
                X_vis = X_vis.detach().cpu().numpy()
            if isinstance(Y_vis, torch.Tensor):
                Y_vis = Y_vis.detach().cpu().numpy()
            x_pts, y_pts = X_vis.flatten(), Y_vis.flatten()
        elif X_pts is not None:
            if isinstance(X_pts, torch.Tensor):
                X_pts = X_pts.detach().cpu().numpy()
            x_pts, y_pts = X_pts[:, 0], X_pts[:, 1]
        else:
            x_pts = y_pts = None
    else:
        x_pts = y_pts = None

    def add_scatter(ax):
        if x_pts is not None and y_pts is not None:
            ax.scatter(x_pts, y_pts, color='black', s=10, alpha=0.8, label='Sample Points')
            ax.legend(loc='upper right')

    def prepare_coverage_data(df):
        expected = 1 - df[alpha_col]
        empirical = df[cov_col]
        exp_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
        emp_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)
        sort_idx = exp_full.argsort()
        exp_sorted, emp_sorted = exp_full[sort_idx], emp_full[sort_idx]
        return exp_sorted.to_numpy(), emp_sorted.to_numpy()

    def coverage_deviation(exp, emp, how="mae"):
        diff = np.abs(emp - exp)
        if how == "mae":
            return diff.mean()
        elif how == "rmse":
            return np.sqrt((diff**2).mean())
        elif how == "max":
            return diff.max()
        else:
            raise ValueError("metric must be 'mae', 'rmse', or 'max'")

    exp1, emp1 = prepare_coverage_data(df_uncal)
    exp2, emp2 = prepare_coverage_data(df_cal)
    dev1 = coverage_deviation(exp1, emp1, metric.lower())
    dev2 = coverage_deviation(exp2, emp2, metric.lower())

    x = XY_test[:, 0].detach().cpu().numpy()
    y = XY_test[:, 1].detach().cpu().numpy()
    x_lin = np.linspace(x.min(), x.max(), grid_size)
    y_lin = np.linspace(y.min(), y.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    XY_grid = torch.tensor(np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1),
                           dtype=XY_test.dtype)

    true_np_grid = true_solution(XY_grid).detach().cpu().numpy().reshape(grid_size, grid_size)

    def prep_interval(pred_set):
        lower = pred_set[0].detach().cpu().numpy().ravel()
        upper = pred_set[1].detach().cpu().numpy().ravel()
        return (lower + upper) / 2.0, upper - lower

    mean_uncal, width_uncal = prep_interval(pred_set_uncal)
    mean_cal, width_cal = prep_interval(pred_set_cal)

    mean_uncal_grid = mean_uncal.reshape(grid_size, grid_size)
    width_uncal_grid = width_uncal.reshape(grid_size, grid_size)
    mean_cal_grid = mean_cal.reshape(grid_size, grid_size)
    width_cal_grid = width_cal.reshape(grid_size, grid_size)

    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    def imshow_plot(ax, data, vlim, title, zoom_factor=4):
        # Smooth upsample: shape (H, W) → (H*zoom, W*zoom)
        data_hr = zoom(data, zoom=zoom_factor, order=3)  # bicubic interpolation

        im = ax.imshow(
            data_hr,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            aspect='auto',
            interpolation='nearest',  # no need for imshow interp; data already smooth
            vmin=None if vlim is None else vlim[0],
            vmax=None if vlim is None else vlim[1],
            cmap='viridis'
        )
        fig.colorbar(im, ax=ax)
        add_scatter(ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Row 0 (Uncalibrated)
    imshow_plot(axs[0, 0], true_np_grid, vlim_true, f"{title}: True u(x,y)")
    imshow_plot(axs[0, 1], mean_uncal_grid, vlim_pred_mean, "Predicted Mean (Uncalibrated)")
    imshow_plot(axs[0, 2], width_uncal_grid, vlim_pred_width, "Interval Width (Uncalibrated)")
    axs[0, 3].plot(exp1, emp1, marker='o', label='Empirical')
    axs[0, 3].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y=x)')
    # axs[0, 3].set_title(f"Coverage (Uncalibrated)\n{metric.upper()}={dev1:.3f}")
    axs[0, 3].set_title(f"Coverage (Uncalibrated)")
    axs[0, 3].set_xlabel("Expected Coverage (1 − α)")
    axs[0, 3].set_ylabel("Empirical Coverage")
    axs[0, 3].legend()
    axs[0, 3].grid(True)

    # Row 1 (Calibrated)
    imshow_plot(axs[1, 0], true_np_grid, vlim_true, f"{title}: True u(x,y)")
    imshow_plot(axs[1, 1], mean_cal_grid, vlim_pred_mean, "Predicted Mean (Calibrated)")
    imshow_plot(axs[1, 2], width_cal_grid, vlim_pred_width, "Interval Width (Calibrated)")
    axs[1, 3].plot(exp2, emp2, marker='o', label='Empirical')
    axs[1, 3].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y=x)')
    # axs[1, 3].set_title(f"Coverage (Calibrated)\n{metric.upper()}={dev2:.3f}")
    axs[1, 3].set_title(f"Coverage (Calibrated)")
    axs[1, 3].set_xlabel("Expected Coverage (1 − α)")
    axs[1, 3].set_ylabel("Empirical Coverage")
    axs[1, 3].legend()
    axs[1, 3].grid(True)

    if main_title is not None:
        fig.suptitle(main_title, fontsize=18, y=1.02)

    plt.tight_layout()







def plot_predictions_2D_compare(
    XY_test,
    pred_set_uncal,
    pred_set_cal,
    true_solution,
    X_pts=None,
    title="2D UQ Result",
    vlim_true=None,
    vlim_pred_mean=None,
    vlim_pred_width=None,
    show_pts=False  # Control whether to scatter X_pts
):
    """
    Plots 2D prediction comparison with optional fixed color ranges and overlay of sample points.

    Args:
        XY_test: Tensor of shape (N, 2)
        pred_set_uncal: Tuple (lower, upper) from uncalibrated model
        pred_set_cal: Tuple (lower, upper) from calibrated model
        true_solution: Callable true u(x,y)
        X_pts: Optional tensor/array of shape (M, 2) — points to overlay
        vlim_true: tuple (vmin, vmax) or None — color range for true u(x,y)
        vlim_pred_mean: tuple (vmin, vmax) or None — color range for predicted means
        vlim_pred_width: tuple (vmin, vmax) or None — color range for interval widths
        show_pts: bool — whether to show X_pts on plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Convert X_pts to numpy and extract x/y
    if X_pts is not None and show_pts:
        if isinstance(X_pts, torch.Tensor):
            X_pts = X_pts.detach().cpu().numpy()
        x_pts = X_pts[:, 0]
        y_pts = X_pts[:, 1]
    else:
        x_pts = y_pts = None

    # Extract test points
    x = XY_test[:, 0].detach().cpu().numpy()
    y = XY_test[:, 1].detach().cpu().numpy()

    # Generate grid for true solution
    grid_size = 100
    x_lin = np.linspace(x.min(), x.max(), grid_size)
    y_lin = np.linspace(y.min(), y.max(), grid_size)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    XY_grid = torch.tensor(np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1), dtype=XY_test.dtype)

    # True solution on grid
    true_np_grid = true_solution(XY_grid).detach().cpu().numpy().reshape(grid_size, grid_size)

    # Helper for prediction intervals
    def prepare(pred_set):
        lower_np = pred_set[0].detach().cpu().numpy().flatten()
        upper_np = pred_set[1].detach().cpu().numpy().flatten()
        mean_np = (lower_np + upper_np) / 2.0
        width_np = upper_np - lower_np
        return mean_np, width_np

    mean_uncal, width_uncal = prepare(pred_set_uncal)
    mean_cal, width_cal = prepare(pred_set_cal)

    # Plot setup
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Function to add scatter if needed
    def add_scatter(ax):
        if x_pts is not None and y_pts is not None:
            ax.scatter(x_pts, y_pts, color='black', s=10, alpha=0.8, label='Sample Points')
            ax.legend(loc='upper right')

    # ─────────────── True Solution ───────────────
    for row in range(2):
        ax = axs[row, 0]
        im = ax.contourf(
            X_grid, Y_grid, true_np_grid, levels=100,
            vmin=None if vlim_true is None else vlim_true[0],
            vmax=None if vlim_true is None else vlim_true[1]
        )
        add_scatter(ax)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{title}: True u(x,y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # ─────────────── Uncalibrated ───────────────
    ax = axs[0, 1]
    im = ax.tricontourf(
        x, y, mean_uncal, levels=100,
        vmin=None if vlim_pred_mean is None else vlim_pred_mean[0],
        vmax=None if vlim_pred_mean is None else vlim_pred_mean[1]
    )
    add_scatter(ax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}: Predicted Mean (Uncalibrated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axs[0, 2]
    im = ax.tricontourf(
        x, y, width_uncal, levels=100,
        vmin=None if vlim_pred_width is None else vlim_pred_width[0],
        vmax=None if vlim_pred_width is None else vlim_pred_width[1]
    )
    add_scatter(ax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}: Interval Width (Uncalibrated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # ─────────────── Calibrated ───────────────
    ax = axs[1, 1]
    im = ax.tricontourf(
        x, y, mean_cal, levels=100,
        vmin=None if vlim_pred_mean is None else vlim_pred_mean[0],
        vmax=None if vlim_pred_mean is None else vlim_pred_mean[1]
    )
    add_scatter(ax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}: Predicted Mean (Calibrated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax = axs[1, 2]
    im = ax.tricontourf(
        x, y, width_cal, levels=100,
        vmin=None if vlim_pred_width is None else vlim_pred_width[0],
        vmax=None if vlim_pred_width is None else vlim_pred_width[1]
    )
    add_scatter(ax)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{title}: Interval Width (Calibrated)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()


def plot_1D_comparison_with_coverage(
    X_test,
    pred_set_uncal,
    pred_set_cal,
    true_solution,
    df_uncal,
    df_cal,
    X_vis=None,
    Y_vis=None,
    title="1D UQ Result",
    vlim_true=None,
    vlim_pred_mean=None,
    vlim_pred_width=None,
    show_pts=True,
    alpha_col='alpha',
    cov_col='coverage',
    metric: str = "mae",
    main_title="1D Prediction with Coverage"
):
    """
    Plots a 1D true solution, predictive intervals (uncalibrated and calibrated), and coverage comparison.

    Parameters
    ----------
    X_test : torch.Tensor
        Test input grid for prediction (1D).
    pred_set_uncal, pred_set_cal : list[torch.Tensor, torch.Tensor]
        Lower and upper bounds of predictive intervals (uncalibrated and calibrated).
    true_solution : callable
        Ground truth function for solution.
    df_uncal, df_cal : pd.DataFrame
        Coverage diagnostics for uncalibrated and calibrated models.
    X_vis, Y_vis : torch.Tensor, optional
        Optional visualization points and their true solution values.
    show_pts : bool
        Whether to show X_vis and Y_vis points.
    metric : str
        Metric to summarize coverage deviation ('mae', 'rmse', or 'max').
    """

    # ─── Prepare input arrays ───────────────────────────────────────────
    x_np = X_test.detach().cpu().numpy().flatten()

    def prep_interval(pred_set):
        lower = pred_set[0].detach().cpu().numpy().flatten()
        upper = pred_set[1].detach().cpu().numpy().flatten()
        return (lower + upper) / 2.0, upper - lower, lower, upper

    mean_uncal, width_uncal, lower_uncal, upper_uncal = prep_interval(pred_set_uncal)
    mean_cal, width_cal, lower_cal, upper_cal = prep_interval(pred_set_cal)

    y_true = true_solution(x_np)

    # ─── Coverage data preparation ──────────────────────────────────────
    def prepare_coverage_data(df):
        expected = 1 - df[alpha_col]
        empirical = df[cov_col]
        exp_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
        emp_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)
        sort_idx = exp_full.argsort()
        exp_sorted, emp_sorted = exp_full[sort_idx], emp_full[sort_idx]
        return exp_sorted.to_numpy(), emp_sorted.to_numpy()

    def coverage_deviation(exp, emp, how="mae"):
        diff = np.abs(emp - exp)
        if   how == "mae":  return diff.mean()
        elif how == "rmse": return np.sqrt((diff**2).mean())
        elif how == "max":  return diff.max()
        else:
            raise ValueError("metric must be 'mae', 'rmse', or 'max'")

    exp1, emp1 = prepare_coverage_data(df_uncal)
    exp2, emp2 = prepare_coverage_data(df_cal)
    dev1 = coverage_deviation(exp1, emp1, metric.lower())
    dev2 = coverage_deviation(exp2, emp2, metric.lower())

    # ─── Prepare visualization points (optional) ────────────────────────
    if X_vis is not None and show_pts:
        print("Plotting the points")
        X_vis_np = X_vis.detach().cpu().numpy().flatten()
        Y_vis_np = Y_vis.detach().cpu().numpy().flatten() if Y_vis is not None else np.zeros_like(X_vis_np)
    else:
        X_vis_np = Y_vis_np = None

    # ─── Plotting ──────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # --- row 0 : Uncalibrated -----------------------------------------
    axs[0, 0].plot(x_np, y_true, 'k--', lw=2, label='True solution')
    axs[0, 0].plot(x_np, mean_uncal, 'b-', lw=2, label='Predicted mean')
    axs[0, 0].fill_between(x_np, lower_uncal, upper_uncal, color='blue', alpha=0.3, label='Confidence interval')

    if X_vis_np is not None:
        axs[0, 0].scatter(X_vis_np, Y_vis_np, c='purple', s=30, alpha=0.7, marker='x', label='Test Points')

    axs[0, 0].set_title("Predicted Mean & Interval (Uncalibrated)")
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('u(x)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(exp1, emp1, marker='o', label='Empirical')
    axs[0, 1].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y=x)')
    axs[0, 1].set_title(f"Coverage (Uncalibrated)\n{metric.upper()}={dev1:.3f}")
    axs[0, 1].set_xlabel("Expected Coverage (1 − α)")
    axs[0, 1].set_ylabel("Empirical Coverage")
    axs[0, 1].legend(); axs[0, 1].grid(True)

    # --- row 1 : Calibrated -------------------------------------------
    axs[1, 0].plot(x_np, y_true, 'k--', lw=2, label='True solution')
    axs[1, 0].plot(x_np, mean_cal, 'g-', lw=2, label='Predicted mean')
    axs[1, 0].fill_between(x_np, lower_cal, upper_cal, color='green', alpha=0.3, label='Confidence interval')

    if X_vis_np is not None:
        axs[1, 0].scatter(X_vis_np, Y_vis_np, c='purple', s=30, alpha=0.7, marker='x', label='Test Points')

    axs[1, 0].set_title("Predicted Mean & Interval (Calibrated)")
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('u(x)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(exp2, emp2, marker='o', label='Empirical')
    axs[1, 1].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y=x)')
    axs[1, 1].set_title(f"Coverage (Calibrated)\n{metric.upper()}={dev2:.3f}")
    axs[1, 1].set_xlabel("Expected Coverage (1 − α)")
    axs[1, 1].set_ylabel("Empirical Coverage")
    axs[1, 1].legend(); axs[1, 1].grid(True)

    if main_title is not None:
        fig.suptitle(main_title, fontsize=18, y=1.02)

    plt.tight_layout()
    # plt.show()



import matplotlib.pyplot as plt

def plot_truth_and_samples_2D(
    X_train, Y_train, grid, U_true_grid, domain,
    title="2D PDE samples visualization"
):
    fig, ax = plt.subplots(figsize=(7, 6))
    # Show the true solution as a colormap
    x = np.linspace(domain[0][0], domain[0][1], U_true_grid.shape[0])
    y = np.linspace(domain[1][0], domain[1][1], U_true_grid.shape[1])
    im = ax.imshow(
        U_true_grid,
        extent=(domain[0][0], domain[0][1], domain[1][0], domain[1][1]),
        origin='lower',
        aspect='auto',
        alpha=0.8,
        cmap='coolwarm'
    )
    # Overlay noisy training points
    ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, edgecolor='k', cmap='viridis', s=18, label="Noisy samples")
    plt.colorbar(im, ax=ax, label="u(x, y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()



import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics_table(
    X_test: torch.Tensor,
    cp_uncal_predset,
    cp_cal_predset,
    true_solution,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str="Uncalibrated",
    df2_name: str="Calibrated",
    title: str = "",
    main_title: str | None = None,
    X_vis=None, Y_vis=None,
    alpha_level: float = 0.05,
    figsize: tuple = (9, 2.5),
    max_digits_display = lambda x: f"{x:.4g}"
):
    """
    Display a side-by-side metrics comparison (table) for the uncalibrated and
    calibrated models at a single alpha level.
    """
    # Compute the coverage deviation using mae
    def prepare_coverage_data(df):
        expected = 1 - df["alpha"]
        empirical = df["coverage"]
        exp_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
        emp_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)
        sort_idx = exp_full.argsort()
        exp_sorted, emp_sorted = exp_full[sort_idx], emp_full[sort_idx]
        return exp_sorted.to_numpy(), emp_sorted.to_numpy()

    def coverage_deviation(exp, emp, how="mae"):
        diff = np.abs(emp - exp)
        if   how == "mae":  return diff.mean()
        elif how == "rmse": return np.sqrt((diff**2).mean())
        elif how == "max":  return diff.max()
        else:
            raise ValueError("metric must be 'mae', 'rmse', or 'max'")

    exp1, emp1 = prepare_coverage_data(df1)
    exp2, emp2 = prepare_coverage_data(df2)
    dev1 = coverage_deviation(exp1, emp1)  # Using the default metrics
    dev2 = coverage_deviation(exp2, emp2)  # Using the default metrics
    print(f"Uncal dev:{dev1}")
    print(f"Cal dev:{dev2}")
    alpha_level_upper = alpha_level + 1e-3
    alpha_level_lower = alpha_level - 1e-3
    
    # ────────────────────── 1. Slice the two rows ──────────────────────
    row_uncal = df1.loc[(df1["alpha"] <= alpha_level_upper) & 
                           (df1["alpha"] >= alpha_level_lower)].copy()
    row_uncal["model"] = df1_name
    row_uncal["expected coverage"] = (1 - row_uncal["alpha"])
    row_uncal["mean coverage deviation"] = "{:.4f}".format(dev1)
    row_uncal["coverage"] = (row_uncal["coverage"]).map("{:.2f}".format)

    row_cal = df2.loc[(df2["alpha"] <= alpha_level_upper) & 
                        (df2["alpha"] >= alpha_level_lower)].copy()
    row_cal["model"] = df2_name
    row_cal["expected coverage"] = (1- row_cal["alpha"])
    row_cal["mean coverage deviation"] = "{:.4f}".format(dev2)
    row_cal["coverage"] = (row_cal["coverage"]).map("{:.2f}".format)

    if row_uncal.empty or row_cal.empty:
        raise ValueError(f"alpha={alpha_level} not found in both data frames.")

    # ───────────────────── 2. Stack & tidy up ──────────────────────────
    rows = pd.concat([row_uncal, row_cal], axis=0).reset_index(drop=True)
    rows = rows.rename(columns={"coverage": "actual coverage"})
    # Get all columns except 'model' for the selection
    other_cols = [c for c in rows.columns if c != "model"]
    rows = rows.loc[:, ["model"] + other_cols]

    
    # nice ordering: model | expected alpha | true alpha | <metrics…>
    metric_cols = [c for c in rows.columns if c not in ("model", "expected coverage", "actual coverage", "mean coverage deviation", "sharpness")]
    rows = rows[["model", "expected coverage", "actual coverage", "mean coverage deviation", "sharpness"]]
    

    # ──────────────── 2.5. Format numeric values ───────────────────────
    # Format all numeric columns to 4 decimal places (excluding 'model' column)
    for col in rows.columns:
        if pd.api.types.is_numeric_dtype(rows[col]):
            rows[col] = rows[col].apply(max_digits_display)  # .4g gives up to 4 significant 

    # ───────────────────── 3. Plot as table ────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=rows.values,
        colLabels=rows.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    if main_title is not None:
        plt.title(main_title, pad=20, fontsize=12)

    plt.tight_layout()


def _to_1d_np(arr):
    """Convert list / ndarray / torch tensor to 1-D NumPy."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr).reshape(-1)


import numpy as np
import matplotlib.pyplot as plt

def plot_1d_intervals_comparison(
    X_test,
    uncal_interval,                  # (n_lower, n_upper)
    cp_intervals,                    # (cp_lower, cp_upper)
    true_solution,                   # array OR callable f(X_test)
    uncal_interval_label="Before CP",
    cal_interval_label="After CP",
    t_train=None,
    y_train=None,
    title="PINN Prediction with Conformal & Naïve Intervals",
    figsize=(7, 5),
    colors=None,                     # optional dict to override colors by key
    alpha_cp=1.0,
    alpha_naive=1.0,
    dot_size=20,
    dot_edge_size=0.5,

):
    """
    Matches the styling/semantics of the plotting script:
      - CP-band mean is drawn as dashed prediction line
      - Legend at lower center with compact handles
      - Labeling uses x / u(x), fontsize=14, ylabel rotation=0
      - Same color palette and scatter edge styling
    """

    # --- default palette (same constants as your script) ---
    palette = {
        "COL_NAIV": "#f7c5c8",
        "COL_MEAN": "#b13c32",
        "COL_CP":   "#abccf4",
        "COL_TRUE": "#222222",
        "COL_SCAT": "#f6d09f",
        "COL_EDGE": "#222222",
    }
    if colors:
        palette.update(colors)
    COL_NAIV = palette["COL_NAIV"]
    COL_MEAN = palette["COL_MEAN"]
    COL_CP   = palette["COL_CP"]
    COL_TRUE = palette["COL_TRUE"]
    COL_SCAT = palette["COL_SCAT"]
    COL_EDGE = palette["COL_EDGE"]

    # --- evaluate truth if callable ---
    if callable(true_solution):
        true_vals = np.asarray(true_solution(X_test)).ravel()
    else:
        true_vals = np.asarray(true_solution).ravel()

    # --- to 1D arrays ---
    x = np.asarray(X_test).ravel()
    n_lower, n_upper = [np.asarray(a).ravel() for a in uncal_interval]
    cp_lower, cp_upper = [np.asarray(a).ravel() for a in cp_intervals]

    # CP mean (to match your script)
    pred_mean = (cp_lower + cp_upper) / 2.0

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # wide -> narrow bands first for nicer layering
    ax.fill_between(x, n_lower, n_upper, color=COL_NAIV, alpha=alpha_naive,
                    label=uncal_interval_label, zorder=2)
    ax.fill_between(x, cp_lower, cp_upper, color=COL_CP, alpha=alpha_cp,
                    label=cal_interval_label, zorder=1)

    # mean + truth
    ax.plot(x, pred_mean, ls="--", lw=2.0, color=COL_MEAN,
            label=r"Prediction", zorder=5)
    ax.plot(x, true_vals, lw=2.4, color=COL_TRUE,
            label=r"True", zorder=3)

    # training data (optional)
    if t_train is not None and y_train is not None:
        ax.scatter(np.asarray(t_train).ravel(), np.asarray(y_train).ravel(),
                   s=dot_size, facecolor=COL_SCAT, edgecolors=COL_EDGE,
                   linewidth=dot_edge_size, label="Data", zorder=4)

    # labels to match script
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$u(x)$", fontsize=14, rotation=0)

    # layout/styling to match script
    ax.margins(x=0)
    ax.legend(loc="lower center", handlelength=1.6, borderpad=0.6)
    ax.set_title(title)
    fig.tight_layout()

    return fig, ax



import numpy as np
import torch
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm  # optional if you want to use `norm`

def plot_uq_field(
    XY_grid: torch.Tensor,
    true_solution,                 # callable: (N,2) torch -> (N,) or (N,1)
    pred_set=None,                 # tuple(lower, upper) torch tensors (optional if mode='true')
    *,
    mode: str = "mean",            # 'true' | 'mean' | 'width'
    grid_shape=None,               # (H, W); if None, inferred as square
    extent=None,                   # [xmin, xmax, ymin, ymax]; if None, inferred from XY_grid
    vlim=None,                     # (vmin, vmax) or None
    scatter=None,                  # (x_pts, y_pts) numpy arrays or None
    cmap="viridis",
    figsize=(6, 5),
    colorbar_label=None,
    cbar_kwargs=None,              # <- NEW: dict passed to fig.colorbar
    norm=None,                     # <- NEW: matplotlib.colors.Normalize (overrides vlim)
    title="",

):
    """
    Plot a single 2D field for: the true solution ('true'), predictive mean ('mean'),
    or predictive interval width ('width').

    Use `vlim=(vmin, vmax)` or `norm=...` to keep multiple plots on the same color scale.
    Customize the color legend via `cbar_kwargs`, e.g. {'ticks': np.linspace(...), 'format': '%.2f'}.
    """
    # ---- infer grid shape ----
    N = XY_grid.shape[0]
    if grid_shape is None:
        side = int(round(np.sqrt(N)))
        if side * side != N:
            raise ValueError(
                "grid_shape not provided and XY_grid is not square-lengthed; "
                f"N={N} cannot form H×W with H=W."
            )
        H = W = side
    else:
        H, W = grid_shape
        if H * W != N:
            raise ValueError(f"grid_shape {grid_shape} incompatible with XY_grid length {N}.")

    # ---- extent from XY_grid if needed ----
    x = XY_grid[:, 0].detach().cpu().numpy()
    y = XY_grid[:, 1].detach().cpu().numpy()
    if extent is None:
        extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    # ---- build the field to plot ----
    mode = mode.lower()
    if mode == "true":
        with torch.no_grad():
            z = true_solution(XY_grid).detach().cpu().numpy().reshape(-1)
        field = z.reshape(H, W)
        default_label = "u(x, y)"
    elif mode in ("mean", "width"):
        if pred_set is None or len(pred_set) != 2:
            raise ValueError("pred_set=(lower, upper) is required for mode 'mean' or 'width'.")
        lower = pred_set[0].detach().cpu().numpy().reshape(-1)
        upper = pred_set[1].detach().cpu().numpy().reshape(-1)
        if mode == "mean":
            field = ((lower + upper) / 2.0).reshape(H, W)
            default_label = "Predictive mean"
        else:  # 'width'
            field = (upper - lower).reshape(H, W)
            default_label = "Interval width"
    else:
        raise ValueError("mode must be 'true', 'mean', or 'width'.")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)
    # If norm is provided, don't pass vmin/vmax (matplotlib will warn if both are given)
    imshow_kwargs = dict(
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        if vlim is not None:
            imshow_kwargs["vmin"], imshow_kwargs["vmax"] = vlim

    im = ax.imshow(field, **imshow_kwargs)

    # Colorbar with customizable legend
    cbar = fig.colorbar(im, ax=ax, **(cbar_kwargs or {}))
    # Only set label if user didn't pass one via cbar_kwargs
    if not (cbar_kwargs and ("label" in cbar_kwargs)):
        cbar.set_label(colorbar_label or default_label)

    if scatter is not None and scatter[0] is not None and scatter[1] is not None:
        ax.scatter(scatter[0], scatter[1], s=10, c="black", alpha=0.8, label="Sample Points")
        ax.legend(loc="upper right")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        title
    )
    fig.tight_layout()
    return fig, ax, field
