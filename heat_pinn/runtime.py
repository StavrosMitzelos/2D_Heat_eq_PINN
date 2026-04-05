from __future__ import annotations

"""Βοηθητικά για runtime, device, seeding και checkpoints"""

import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, PathsConfig, build_baseline_run_stem


def set_seed(seed: int) -> None:
    """Ορίζει κοινό seed για reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer_directml: bool = True) -> tuple[torch.device, str]:
    """Διαλέγει DirectML αν υπάρχει, αλλιώς CPU"""

    if prefer_directml:
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "directml"
        except ImportError:
            pass

    return torch.device("cpu"), "cpu"


def ensure_output_dirs(paths: PathsConfig) -> None:
    """Φτιάχνει τους βασικούς φακέλους εξόδου αν λείπουν"""

    for directory in (
        paths.runs_dir,
        paths.results_dir,
        paths.results_dir / "grid_searches",
        paths.pruning_sweeps_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_checkpoint_exists(
    checkpoint_path: Path,
    *,
    producer_script: str = "train_heat_pinn.py",
) -> None:
    """Σταματά αν λείπει checkpoint"""

    if checkpoint_path.exists():
        return

    raise FileNotFoundError(
        "Checkpoint not found: "
        f"{checkpoint_path}. "
        f"Run `python {producer_script}` first to generate the baseline checkpoint, "
        "or verify that your current config matches the trained model name."
    )


def resolve_latest_baseline_checkpoint_path(
    config: ExperimentConfig,
    *,
    checkpoint_kind: str = "final",
) -> Path:
    """Βρίσκει το πιο πρόσφατο checkpoint που ταιριάζει στο config"""

    if checkpoint_kind not in {"adam", "final"}:
        raise ValueError("checkpoint_kind must be 'adam' or 'final'.")

    stem = build_baseline_run_stem(config)
    timestamp_pattern = r"\d{8}_\d{6}(?:_\d{3})?"
    if checkpoint_kind == "adam":
        pattern = re.compile(rf"^{re.escape(stem)}_{timestamp_pattern}$")
        checkpoint_name = "baseline_best_adam.pth"
    else:
        pattern = re.compile(rf"^{re.escape(stem)}_{timestamp_pattern}$")
        checkpoint_name = "baseline_final_lbfgs.pth"

    # Ψάχνω στα timestamped runs
    candidates: list[Path] = []
    if config.paths.runs_dir.exists():
        candidates = [
            run_dir / "models" / checkpoint_name
            for run_dir in config.paths.runs_dir.iterdir()
            if run_dir.is_dir()
            and pattern.fullmatch(run_dir.name)
            and (run_dir / "models" / checkpoint_name).exists()
        ]

    if candidates:
        return max(candidates, key=lambda path: path.parent.parent.name)

    requested_kind = "Adam" if checkpoint_kind == "adam" else "final"
    raise FileNotFoundError(
        f"No timestamped {requested_kind} checkpoint found for configuration prefix "
        f"'{stem}' in {config.paths.runs_dir}. Run `python train_heat_pinn.py` first."
    )


def baseline_run_name_from_checkpoint_path(checkpoint_path: Path) -> str:
    """Βγάζει το run name από το path ενός checkpoint"""

    if checkpoint_path.parent.name == "models":
        return checkpoint_path.parent.parent.name
    if checkpoint_path.stem.endswith("_adam"):
        return checkpoint_path.stem[:-5]
    return checkpoint_path.stem


def load_checkpoint(
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> object:
    """Φορτώνει checkpoint από δίσκο"""

    # Τα checkpoints παράγονται τοπικά από το ίδιο project
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def finalize_process(exit_code: int = 0) -> None:
    """Κλείνει καθαρά figures και τερματίζει τη διεργασία"""

    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(exit_code)
