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
    show: bool = True,
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
    ax.scatter(x_colloc_np, [0.0] * len(x_colloc_np), color='red', marker='x', label="Collocation points", alpha=0.6)

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax  # handy when show=False



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

def plot_predictions_1D(x_grid, pred_set, true_solution, title="PDE UQ",
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

    plt.title(title)
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

def plot_dual_expected_vs_empirical(df_uncal, df_cal, 
                                     alpha_col='alpha', cov_col='coverage',
                                     title1='Uncalibrated Model', title2='Calibrated Model'):
    """
    Plots side-by-side coverage plots for uncalibrated and calibrated models.

    Parameters:
    - df_uncal: pd.DataFrame with alpha and coverage for the uncalibrated model
    - df_cal: pd.DataFrame with alpha and coverage for the calibrated model
    - alpha_col: column name for alpha values
    - cov_col: column name for empirical coverage
    - title1: title for the uncalibrated plot
    - title2: title for the calibrated plot
    """

    def prepare_coverage_data(df):
        expected = 1 - df[alpha_col]
        empirical = df[cov_col]
        expected_full = pd.concat([pd.Series([0.0]), expected, pd.Series([1.0])], ignore_index=True)
        empirical_full = pd.concat([pd.Series([0.0]), empirical, pd.Series([1.0])], ignore_index=True)
        sorted_idx = expected_full.argsort()
        return expected_full[sorted_idx], empirical_full[sorted_idx]

    expected1, empirical1 = prepare_coverage_data(df_uncal)
    expected2, empirical2 = prepare_coverage_data(df_cal)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot uncalibrated
    axes[0].plot(expected1, empirical1, marker='o', label='Empirical Coverage')
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y = x)')
    axes[0].set_title(title1)
    axes[0].set_xlabel("Expected Coverage (1 − α)")
    axes[0].set_ylabel("Empirical Coverage")
    axes[0].grid(True)
    axes[0].legend()

    # Plot calibrated
    axes[1].plot(expected2, empirical2, marker='o', label='Empirical Coverage')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', label='Ideal (y = x)')
    axes[1].set_title(title2)
    axes[1].set_xlabel("Expected Coverage (1 − α)")
    axes[1].set_ylabel("Empirical Coverage")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


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