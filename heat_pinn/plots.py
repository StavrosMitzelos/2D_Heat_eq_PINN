from __future__ import annotations

"""Graphs για training, αξιολόγηση και pruning"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DomainConfig
from .data import TrainingData
from .problem import exact_solution, mae, relative_l2_error, rmse


def _exact_snapshot_limits(
    device: torch.device,
    domain: DomainConfig,
    snapshot_times: tuple[float, ...] | list[float],
    n_plot: int,
) -> tuple[float, float]:
    """Υπολογίζει σταθερά όρια για τα snapshot plots"""

    x_vals = np.linspace(domain.x_min, domain.x_max, n_plot)
    y_vals = np.linspace(domain.y_min, domain.y_max, n_plot)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    value_min = float("inf")
    value_max = float("-inf")
    for fixed_t in snapshot_times:
        t_flat = np.full_like(x_flat, fixed_t)
        x_test = torch.tensor(x_flat, dtype=torch.float32).view(-1, 1).to(device)
        y_test = torch.tensor(y_flat, dtype=torch.float32).view(-1, 1).to(device)
        t_test = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_exact = exact_solution(t_test, x_test, y_test)

        value_min = min(value_min, torch.min(u_exact).item())
        value_max = max(value_max, torch.max(u_exact).item())

    return value_min, value_max


def _exact_spacetime_limits(
    device: torch.device,
    domain: DomainConfig,
    fixed_y: float,
    n_space: int,
    n_time: int,
) -> tuple[float, float]:
    """Υπολογίζει σταθερά όρια για τα x-t plots"""

    x_vals = np.linspace(domain.x_min, domain.x_max, n_space)
    t_vals = np.linspace(domain.t_min, domain.t_max, n_time)
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)

    x_flat = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    y_flat = torch.full_like(x_flat, fixed_y).to(device)

    with torch.no_grad():
        u_exact = exact_solution(t_flat, x_flat, y_flat)

    value_min = torch.min(u_exact).item()
    value_max = torch.max(u_exact).item()
    return value_min, value_max


def _exact_slice_limits(
    device: torch.device,
    domain: DomainConfig,
    times_to_plot: tuple[float, ...] | list[float],
    fixed_y: float,
    n_points: int,
) -> tuple[float, float]:
    """Υπολογίζει σταθερά y-limits για όλα τα slice plots"""

    x_vals = np.linspace(domain.x_min, domain.x_max, n_points)
    x_torch = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1).to(device)
    y_torch = torch.full_like(x_torch, fixed_y).to(device)

    value_min = float("inf")
    value_max = float("-inf")

    for t_val in times_to_plot:
        t_torch = torch.full_like(x_torch, t_val).to(device)
        with torch.no_grad():
            u_exact = exact_solution(t_torch, x_torch, y_torch)

        value_min = min(value_min, torch.min(u_exact).item())
        value_max = max(value_max, torch.max(u_exact).item())

    span = value_max - value_min
    pad = 0.05 * span if span > 0 else 0.05
    return value_min - pad, value_max + pad


def compute_snapshot_error_vmax(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    snapshot_times: tuple[float, ...] | list[float],
    n_plot: int = 100,
) -> float:
    """Υπολογίζει το κοινό vmax για τα error snapshot plots"""

    model.eval()

    x_vals = np.linspace(domain.x_min, domain.x_max, n_plot)
    y_vals = np.linspace(domain.y_min, domain.y_max, n_plot)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    error_vmax = 0.0
    for fixed_t in snapshot_times:
        t_flat = np.full_like(x_flat, fixed_t)

        x_test = torch.tensor(x_flat, dtype=torch.float32).view(-1, 1).to(device)
        y_test = torch.tensor(y_flat, dtype=torch.float32).view(-1, 1).to(device)
        t_test = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = model(t_test, x_test, y_test)
            u_exact = exact_solution(t_test, x_test, y_test)
            abs_error = torch.abs(u_pred - u_exact)

        error_vmax = max(error_vmax, torch.max(abs_error).item())

    return max(error_vmax, 1e-12)


def compute_spacetime_error_vmax(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    fixed_y: float,
    n_space: int = 200,
    n_time: int = 200,
) -> float:
    """Υπολογίζει το κοινό vmax για τα error x-t plots"""

    model.eval()

    x_vals = np.linspace(domain.x_min, domain.x_max, n_space)
    t_vals = np.linspace(domain.t_min, domain.t_max, n_time)
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)

    x_test = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    t_test = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32).to(device)
    y_test = torch.full_like(x_test, fixed_y).to(device)

    with torch.no_grad():
        u_pred = model(t_test, x_test, y_test)
        u_exact = exact_solution(t_test, x_test, y_test)
        abs_error = torch.abs(u_pred - u_exact)

    return max(torch.max(abs_error).item(), 1e-12)


def _finalize_figure(fig: plt.Figure, output_path: Path | None, show: bool) -> None:
    """Αποθηκεύει ή εμφανίζει το figure και μετά το κλείνει"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_points(
    data: TrainingData,
    domain: DomainConfig,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Σχεδιάζει τα σημεία collocation και supervised"""

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        data.x_f.detach().cpu().numpy(),
        data.y_f.detach().cpu().numpy(),
        s=1,
        alpha=0.35,
        label="Collocation points",
    )
    ax.scatter(
        data.x_u.detach().cpu().numpy(),
        data.y_u.detach().cpu().numpy(),
        s=6,
        c="red",
        alpha=0.75,
        label="Initial / Boundary points",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Training points distribution (spatial projection)")
    ax.set_xlim(domain.x_min - 0.05, domain.x_max + 0.05)
    ax.set_ylim(domain.y_min - 0.05, domain.y_max + 0.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_figure(fig, output_path, show)


def plot_training_history(
    history: dict[str, list[float]],
    title: str,
    xlabel: str,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Σχεδιάζει την ιστορία των losses"""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(history["total"], label="Total loss")
    ax.semilogy(history["physics"], label="Physics loss")
    ax.semilogy(history["data"], label="Data loss")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_figure(fig, output_path, show)


def plot_snapshot_grid(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    snapshot_times: tuple[float, ...] | list[float],
    n_plot: int = 100,
    error_vmax: float | None = None,
    output_path: Path | None = None,
    show: bool = False,
) -> pd.DataFrame:
    """Σχεδιάζει predicted, exact και absolute error για κάθε snapshot"""

    model.eval()
    value_vmin, value_vmax = _exact_snapshot_limits(
        device,
        domain,
        snapshot_times,
        n_plot,
    )

    x_vals = np.linspace(domain.x_min, domain.x_max, n_plot)
    y_vals = np.linspace(domain.y_min, domain.y_max, n_plot)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    rows: list[dict[str, float]] = []
    snapshot_cache: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []

    # Κάθε γραμμή έχει predicted, exact και error
    fig, axes = plt.subplots(len(snapshot_times), 3, figsize=(18, 4 * len(snapshot_times)))

    if len(snapshot_times) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, fixed_t in enumerate(snapshot_times):
        # Υπολογίζω predictions και metrics για κάθε χρόνο
        t_flat = np.full_like(x_flat, fixed_t)

        x_test = torch.tensor(x_flat, dtype=torch.float32).view(-1, 1).to(device)
        y_test = torch.tensor(y_flat, dtype=torch.float32).view(-1, 1).to(device)
        t_test = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = model(t_test, x_test, y_test)
            u_exact = exact_solution(t_test, x_test, y_test)
            abs_error = torch.abs(u_pred - u_exact)

        rows.append(
            {
                "t": float(fixed_t),
                "max_u_exact": torch.max(torch.abs(u_exact)).item(),
                "max_u_pred": torch.max(torch.abs(u_pred)).item(),
                "relative_l2": relative_l2_error(u_pred, u_exact),
                "mae": mae(u_pred, u_exact),
                "rmse": rmse(u_pred, u_exact),
                "max_error": torch.max(abs_error).item(),
            }
        )

        u_pred_grid = u_pred.cpu().numpy().reshape(n_plot, n_plot)
        u_exact_grid = u_exact.cpu().numpy().reshape(n_plot, n_plot)
        error_grid = abs_error.cpu().numpy().reshape(n_plot, n_plot)
        snapshot_cache.append((float(fixed_t), u_pred_grid, u_exact_grid, error_grid))

    error_vmax_global = max(error_grid.max() for _, _, _, error_grid in snapshot_cache)
    error_vmax_global = max(error_vmax_global, 1e-12)
    if error_vmax is not None:
        error_vmax_global = max(error_vmax, 1e-12)

    # Αν δοθεί κοινό error_vmax, κρατάω σταθερή κλίμακα για δίκαιη σύγκριση
    for i, (fixed_t, u_pred_grid, u_exact_grid, error_grid) in enumerate(snapshot_cache):
        im1 = axes[i, 0].imshow(
            u_pred_grid,
            extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
            origin="lower",
            cmap="viridis",
            vmin=value_vmin,
            vmax=value_vmax,
        )
        axes[i, 0].set_title(f"Predicted u(t={fixed_t:.2f})", fontweight="bold")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("y")
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].imshow(
            u_exact_grid,
            extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
            origin="lower",
            cmap="viridis",
            vmin=value_vmin,
            vmax=value_vmax,
        )
        axes[i, 1].set_title(f"Exact u(t={fixed_t:.2f})", fontweight="bold")
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("y")
        fig.colorbar(im2, ax=axes[i, 1])

        im3 = axes[i, 2].imshow(
            error_grid,
            extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max],
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=error_vmax_global,
        )
        axes[i, 2].set_title(f"Absolute error at t={fixed_t:.2f}", fontweight="bold")
        axes[i, 2].set_xlabel("x")
        axes[i, 2].set_ylabel("y")
        fig.colorbar(im3, ax=axes[i, 2])

    fig.tight_layout()
    _finalize_figure(fig, output_path, show)
    return pd.DataFrame(rows)


def plot_spacetime_slice_grid(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    fixed_y: float,
    snapshot_times: tuple[float, ...] | list[float],
    n_space: int = 200,
    n_time: int = 200,
    error_vmax: float | None = None,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Σχεδιάζει την εξέλιξη της λύσης σε ένα x-t slice"""

    model.eval()
    value_vmin, value_vmax = _exact_spacetime_limits(
        device,
        domain,
        fixed_y,
        n_space,
        n_time,
    )

    x_vals = np.linspace(domain.x_min, domain.x_max, n_space)
    t_vals = np.linspace(domain.t_min, domain.t_max, n_time)
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)

    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)
    y_flat = np.full_like(x_flat, fixed_y)

    x_test = torch.tensor(x_flat, dtype=torch.float32).to(device)
    t_test = torch.tensor(t_flat, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_flat, dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred = model(t_test, x_test, y_test).cpu().numpy().reshape(n_time, n_space)
        u_exact = exact_solution(t_test, x_test, y_test).cpu().numpy().reshape(n_time, n_space)

    error_grid = np.abs(u_pred - u_exact)
    error_vmax_current = max(error_grid.max(), 1e-12)
    if error_vmax is not None:
        error_vmax_current = max(error_vmax, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    panels = [
        (u_pred, "Predicted x-t slice", "viridis", value_vmin, value_vmax),
        (u_exact, "Exact x-t slice", "viridis", value_vmin, value_vmax),
        (error_grid, "Absolute error x-t slice", "magma", 0.0, error_vmax_current),
    ]

    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            values,
            extent=[domain.x_min, domain.x_max, domain.t_min, domain.t_max],
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        for snapshot_time in snapshot_times:
            ax.axhline(snapshot_time, color="white", linestyle="--", linewidth=1, alpha=0.8)

        ax.set_title(f"{title} at y={fixed_y:.2f}", fontweight="bold")
        ax.set_xlabel("x")
        fig.colorbar(image, ax=ax)

    axes[0].set_ylabel("t")
    fig.tight_layout()
    _finalize_figure(fig, output_path, show)


def plot_slice_comparison(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    times_to_plot: tuple[float, ...] | list[float],
    fixed_y: float,
    n_points: int = 200,
    output_path: Path | None = None,
    show: bool = False,
    model_label: str = "PINN",
) -> None:
    """Σχεδιάζει τομή της λύσης σε σταθερό y"""

    model.eval()
    y_min, y_max = _exact_slice_limits(
        device,
        domain,
        times_to_plot,
        fixed_y,
        n_points,
    )

    x_vals = np.linspace(domain.x_min, domain.x_max, n_points)
    x_torch = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1).to(device)
    y_torch = torch.full_like(x_torch, fixed_y).to(device)

    fig = plt.figure(figsize=(4 * len(times_to_plot), 5))
    for i, t_val in enumerate(times_to_plot):
        t_torch = torch.full_like(x_torch, t_val).to(device)

        with torch.no_grad():
            u_pred = model(t_torch, x_torch, y_torch).cpu().numpy()
            u_exact = exact_solution(t_torch, x_torch, y_torch).cpu().numpy()

        ax = plt.subplot(1, len(times_to_plot), i + 1)
        ax.plot(x_vals, u_exact, "b-", linewidth=2, label="Exact")
        ax.plot(x_vals, u_pred, "r--", linewidth=2, label=model_label)
        ax.set_title(f"t = {t_val:.2f}", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_xlim(domain.x_min, domain.x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.6)

        if i == 0:
            ax.set_ylabel(f"u(t, x, y={fixed_y})")
            ax.legend(loc="upper right")

    fig.suptitle("Slice comparison: exact vs PINN", fontsize=16)
    fig.tight_layout()
    _finalize_figure(fig, output_path, show)


def plot_sweep_curve(
    sweep_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Σχεδιάζει καμπύλη αποτελεσμάτων από sweep pruning"""

    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_subset = sweep_df[sweep_df["pruning_type"] == "none"].copy()
    ax.scatter(baseline_subset[x_col], baseline_subset[y_col], label="baseline")

    for pruning_type in ("unstructured", "structured"):
        subset = sweep_df[sweep_df["pruning_type"] == pruning_type].copy()
        subset = subset.sort_values(
            x_col,
            ascending=x_col == "final_sparsity_percent",
        )
        ax.plot(subset[x_col], subset[y_col], marker="o", label=pruning_type)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize_figure(fig, output_path, show)
