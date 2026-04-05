"""Βασικά exports του heat_pinn"""

# Ρυθμίσεις πειράματος
from .config import ExperimentConfig

# Δεδομένα εκπαίδευσης
from .data import TrainingData, generate_training_data

# Συναρτήσεις αξιολόγησης
from .evaluation import (
    evaluate_global_relative_l2,
    evaluate_model_on_snapshots,
    summarize_snapshot_metrics,
)

# Το PINN
from .model import HeatPINN

# Η PDE και οι μετρικές σφάλματος
from .problem import (
    exact_solution,
    mae,
    pde_residual,
    physics_loss_function,
    relative_l2_error,
    rmse,
)

# Τα ονόματα που εκθέτουμε
__all__ = [
    "ExperimentConfig",
    "HeatPINN",
    "TrainingData",
    "evaluate_global_relative_l2",
    "evaluate_model_on_snapshots",
    "exact_solution",
    "generate_training_data",
    "mae",
    "pde_residual",
    "physics_loss_function",
    "relative_l2_error",
    "rmse",
    "summarize_snapshot_metrics",
]
