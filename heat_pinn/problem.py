from __future__ import annotations

"""Η αναλυτική λύση, το residual της PDE και οι μετρικές σφάλματος"""

import numpy as np
import torch


def exact_solution(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ακριβής λύση με μη μηδενική Dirichlet συνθήκη στο πάνω σύνορο"""

    x_unit = 0.5 * (x + 1.0)
    y_unit = 0.5 * (y + 1.0)

    steady_state = torch.sin(np.pi * x_unit) * torch.sinh(np.pi * y_unit) / np.sinh(np.pi)
    transient = (
        torch.exp(-0.5 * np.pi**2 * t)
        * torch.sin(np.pi * x_unit)
        * torch.sin(np.pi * y_unit)
    )
    return steady_state + transient


def pde_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Υπολογίζει το residual u_t - u_xx - u_yy"""

    u = model(t, x, y)

    # Πρώτες και δεύτερες παράγωγοι που χρειάζονται στην PDE
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_y = torch.autograd.grad(
        u,
        y,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]

    u_yy = torch.autograd.grad(
        u_y,
        y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
    )[0]

    return u_t - u_xx - u_yy


def physics_loss_function(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """MSE του residual της PDE"""

    return torch.mean(pde_residual(model, t, x, y) ** 2)


def relative_l2_error(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Relative L2 error μεταξύ πρόβλεψης και ακριβούς λύσης"""

    num = torch.sqrt(torch.sum((u_pred - u_exact) ** 2))
    den = torch.sqrt(torch.sum(u_exact ** 2))
    return (num / (den + eps)).item()


def mae(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    """Mean absolute error"""

    return torch.mean(torch.abs(u_pred - u_exact)).item()


def rmse(u_pred: torch.Tensor, u_exact: torch.Tensor) -> float:
    """Root mean squared error"""

    return torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
