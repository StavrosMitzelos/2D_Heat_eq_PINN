from __future__ import annotations

"""Βοηθητικά για το registry των πειραμάτων"""

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentConfig


def build_experiment_row(
    config: ExperimentConfig,
    model_label: str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Χτίζει μία γραμμή για το registry με τα βασικά στοιχεία του run"""

    row: dict[str, Any] = {
        "model_label": model_label,
        "adam_epochs": config.training.epochs_adam,
        "adam_lr": config.training.adam_lr,
        "lambda_f": config.training.lambda_f,
        "lambda_u": config.training.lambda_u,
        "lbfgs_lr": config.training.lbfgs_lr,
        "lbfgs_max_iter": config.training.lbfgs_max_iter,
        "lbfgs_max_eval": config.training.lbfgs_max_eval,
    }
    row.update(metrics)
    if extra is not None:
        row.update(extra)
    return row


def append_experiment_rows(excel_path: Path, rows: list[dict[str, Any]]) -> None:
    """Προσθέτει νέες γραμμές στο αρχείο registry"""

    if not rows:
        return

    excel_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)
    if excel_path.exists():
        existing_df = pd.read_excel(excel_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined_df = new_df

    combined_df.to_excel(excel_path, index=False, sheet_name="Experiments")
