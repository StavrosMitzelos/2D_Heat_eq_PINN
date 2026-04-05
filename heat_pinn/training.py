from __future__ import annotations

"""Εκπαίδευση του PINN με Adam και L-BFGS"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .config import TrainingConfig
from .data import TrainingData
from .problem import physics_loss_function

try:
    from tqdm.auto import tqdm, trange
except ImportError:  # pragma: no cover
    tqdm = None
    trange = None


@dataclass
class AdamTrainingResult:
    """Αποτέλεσμα της φάσης Adam"""

    history: dict[str, list[float]]
    best_loss: float
    best_epoch: int
    checkpoint_path: Path | None


@dataclass
class LBFGSTrainingResult:
    """Αποτέλεσμα της φάσης L-BFGS"""

    history: dict[str, list[float]]
    best_loss: float
    best_step: int
    checkpoint_path: Path | None

def compute_total_loss(
    model: torch.nn.Module,
    data: TrainingData,
    lambda_f: float,
    lambda_u: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Υπολογίζει total, physics και data loss"""

    loss_physics = physics_loss_function(model, data.t_f, data.x_f, data.y_f)
    u_pred = model(data.t_u, data.x_u, data.y_u)
    loss_data = torch.mean((u_pred - data.u_real) ** 2)
    loss = lambda_f * loss_physics + lambda_u * loss_data
    return loss, loss_physics, loss_data


def train_with_adam(
    model: torch.nn.Module,
    data: TrainingData,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
) -> AdamTrainingResult:
    """Εκπαιδεύει το μοντέλο με Adam και κρατά το καλύτερο state"""

    optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)

    history = {"total": [], "physics": [], "data": []}
    best_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    iterator = (
        trange(1, config.epochs_adam + 1, desc="Adam")
        if trange is not None
        else range(1, config.epochs_adam + 1)
    )

    model.train()
    for epoch in iterator:
        # Κλασικό βήμα Adam
        optimizer.zero_grad()
        loss, loss_physics, loss_data = compute_total_loss(
            model,
            data,
            config.lambda_f,
            config.lambda_u,
        )
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        history["total"].append(loss_value)
        history["physics"].append(loss_physics.item())
        history["data"].append(loss_data.item())

        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if trange is not None and epoch % 100 == 0:
            iterator.set_postfix(
                total=f"{loss_value:.3e}",
                phys=f"{loss_physics.item():.3e}",
                data=f"{loss_data.item():.3e}",
            )

    if best_state_dict is None:
        raise RuntimeError("Adam training did not produce a valid model state.")

    model.load_state_dict(best_state_dict)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return AdamTrainingResult(
        history=history,
        best_loss=best_loss,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
    )


def make_lbfgs_closure(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TrainingData,
    lambda_f: float,
    lambda_u: float,
    desc: str = "L-BFGS",
    total_steps: int | None = None,
) -> tuple[Callable[[], torch.Tensor], dict[str, list[float]], object | None, dict[str, object]]:
    """Φτιάχνει closure και tracker για το L-BFGS"""

    history = {"total": [], "physics": [], "data": []}
    progress = tqdm(total=total_steps, desc=desc) if tqdm is not None and total_steps is not None else None
    closure_calls = 0
    tracker: dict[str, object] = {
        "best_loss": float("inf"),
        "best_step": 0,
        "best_state_dict": None,
    }

    def closure() -> torch.Tensor:
        # Το L-BFGS ζητά closure που ξαναϋπολογίζει loss και gradients
        nonlocal closure_calls
        optimizer.zero_grad()
        loss, loss_physics, loss_data = compute_total_loss(
            model,
            data,
            lambda_f,
            lambda_u,
        )
        loss.backward()

        loss_value = loss.item()
        history["total"].append(loss_value)
        history["physics"].append(loss_physics.item())
        history["data"].append(loss_data.item())

        closure_calls += 1
        if loss_value < tracker["best_loss"]:
            tracker["best_loss"] = loss_value
            tracker["best_step"] = closure_calls
            tracker["best_state_dict"] = copy.deepcopy(model.state_dict())

        if progress is not None:
            if progress.total is not None and closure_calls > progress.total:
                progress.total = closure_calls
            progress.update(1)
            progress.set_postfix(
                total=f"{loss_value:.3e}",
                phys=f"{loss_physics.item():.3e}",
                data=f"{loss_data.item():.3e}",
            )
        return loss

    return closure, history, progress, tracker


def fine_tune_with_lbfgs(
    model: torch.nn.Module,
    data: TrainingData,
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
    max_iter: int | None = None,
    max_eval: int | None = None,
) -> LBFGSTrainingResult:
    """Κάνει fine-tuning με L-BFGS και κρατά το καλύτερο state"""

    step_limit = max_eval or config.lbfgs_max_eval
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=config.lbfgs_lr,
        max_iter=max_iter or config.lbfgs_max_iter,
        max_eval=step_limit,
        history_size=config.lbfgs_history_size,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    closure, history, progress, tracker = make_lbfgs_closure(
        model,
        optimizer,
        data,
        config.lambda_f,
        config.lambda_u,
        total_steps=step_limit,
    )

    model.train()
    try:
        optimizer.step(closure)
    finally:
        if progress is not None:
            progress.close()
        else:
            print(f"L-BFGS evaluations recorded: {len(history['total'])}")

    best_state_dict = tracker["best_state_dict"]
    if best_state_dict is None:
        raise RuntimeError("L-BFGS fine-tuning did not produce a valid model state.")

    model.load_state_dict(best_state_dict)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return LBFGSTrainingResult(
        history=history,
        best_loss=float(tracker["best_loss"]),
        best_step=int(tracker["best_step"]),
        checkpoint_path=checkpoint_path,
    )
