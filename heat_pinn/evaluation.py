from __future__ import annotations

"""Συναρτήσεις αξιολόγησης του PINN"""

import numpy as np
import pandas as pd
import torch

from .config import DomainConfig
from .problem import exact_solution, mae, relative_l2_error, rmse


def evaluate_model_on_snapshots(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    snapshot_times: tuple[float, ...] | list[float],
    n_plot: int = 100,
) -> pd.DataFrame:
    """Υπολογίζει μετρικές σε συγκεκριμένες χρονικές στιγμές"""

    model.eval()

    x_vals = np.linspace(domain.x_min, domain.x_max, n_plot)
    y_vals = np.linspace(domain.y_min, domain.y_max, n_plot)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    rows: list[dict[str, float]] = []
    for fixed_t in snapshot_times:
        # Χτίζω το πλέγμα για ένα σταθερό t
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
                "relative_l2": relative_l2_error(u_pred, u_exact),
                "mae": mae(u_pred, u_exact),
                "rmse": rmse(u_pred, u_exact),
                "max_error": torch.max(abs_error).item(),
            }
        )

    return pd.DataFrame(rows)


def summarize_snapshot_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Βγάζει μέσες και χειρότερες τιμές από τα snapshot metrics"""

    return {
        "mean_relative_l2": float(df["relative_l2"].mean()),
        "mean_mae": float(df["mae"].mean()),
        "mean_rmse": float(df["rmse"].mean()),
        "mean_max_error": float(df["max_error"].mean()),
        "worst_relative_l2": float(df["relative_l2"].max()),
        "worst_mae": float(df["mae"].max()),
        "worst_rmse": float(df["rmse"].max()),
        "worst_max_error": float(df["max_error"].max()),
    }


def evaluate_global_relative_l2(
    model: torch.nn.Module,
    device: torch.device,
    domain: DomainConfig,
    n_space: int = 80,
    n_times: int = 11,
) -> dict[str, float]:
    """Υπολογίζει global relative L2 σε πολλά χρονικά επίπεδα"""

    model.eval()

    x_vals = np.linspace(domain.x_min, domain.x_max, n_space)
    y_vals = np.linspace(domain.y_min, domain.y_max, n_space)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    times = np.linspace(domain.t_min, domain.t_max, n_times)

    rel_l2_values: list[float] = []
    for t_val in times:
        t_flat = np.full_like(x_flat, t_val)

        x_test = torch.tensor(x_flat, dtype=torch.float32).view(-1, 1).to(device)
        y_test = torch.tensor(y_flat, dtype=torch.float32).view(-1, 1).to(device)
        t_test = torch.tensor(t_flat, dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():
            u_pred = model(t_test, x_test, y_test)
            u_exact = exact_solution(t_test, x_test, y_test)

        rel_l2_values.append(relative_l2_error(u_pred, u_exact))

    return {
        "global_mean_relative_l2": float(np.mean(rel_l2_values)),
        "global_worst_relative_l2": float(np.max(rel_l2_values)),
    }
