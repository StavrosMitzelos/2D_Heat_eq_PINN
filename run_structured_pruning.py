from __future__ import annotations

"""Structured pruning experiment για το καλύτερο nonzero-top (μη μηδενική συνθήκη Dirichlet) PINN ."""


import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from heat_pinn.config import ExperimentConfig, derive_lbfgs_max_eval
from heat_pinn.data import generate_training_data
from heat_pinn.evaluation import (
    evaluate_global_relative_l2,
    evaluate_model_on_snapshots,
    summarize_snapshot_metrics,
)
from heat_pinn.model import HeatPINN
from heat_pinn.plots import plot_slice_comparison, plot_snapshot_grid, plot_training_history
from heat_pinn.reporting import write_dataframe_report
from heat_pinn.runtime import ensure_output_dirs, finalize_process, get_device, load_checkpoint, set_seed
from heat_pinn.training import fine_tune_with_lbfgs


PRUNING_AMOUNTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LBFGS_MAX_ITER = 500
LBFGS_MAX_EVAL = derive_lbfgs_max_eval(LBFGS_MAX_ITER)

NONZERO_RESULTS_PATH = Path("results/grid_searches/nonzero_top_dirichlet_sample_runs/results.xlsx")
OUTPUT_ROOT = Path("pruning_sweeps")

def load_best_baseline_row() -> pd.Series:
    """Διαβάζει το καλύτερο nonzero-top baseline από το results.xlsx."""

    if not NONZERO_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Report not found: {NONZERO_RESULTS_PATH}")

    results_df = pd.read_excel(NONZERO_RESULTS_PATH)
    completed_df = results_df[results_df["status"] == "completed"].copy()
    if completed_df.empty:
        raise RuntimeError("No completed rows found in nonzero-top results.xlsx.")

    return completed_df.sort_values(
        ["global_mean_relative_l2", "global_worst_relative_l2"],
    ).iloc[0]


def checkpoint_path_from_row(row: pd.Series) -> Path:
    """Παίρνει το checkpoint path από το καλύτερο row."""

    checkpoint_path = Path(str(row["final_checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path

    if not checkpoint_path.exists():
        run_name = str(row["run_name"])
        checkpoint_path = (
            NONZERO_RESULTS_PATH.parent
            / run_name
            / "models"
            / "baseline_final_lbfgs.pth"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def infer_layer_sizes(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    """Συμπεραίνει την αρχιτεκτονική από τα Linear weights."""

    layer_sizes: list[int] = []
    layer_index = 0
    while f"layers.{layer_index}.weight" in state_dict:
        weight = state_dict[f"layers.{layer_index}.weight"]
        if layer_index == 0:
            layer_sizes.append(int(weight.shape[1]))
        layer_sizes.append(int(weight.shape[0]))
        layer_index += 1

    if len(layer_sizes) < 2:
        raise RuntimeError("Could not infer layer sizes from checkpoint.")

    return tuple(layer_sizes)


def load_dense_model(
    checkpoint_path: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[HeatPINN, tuple[int, ...]]:
    """Φορτώνει το dense baseline μοντέλο."""

    state_dict = load_checkpoint(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Baseline checkpoint is not a state_dict.")

    layer_sizes = infer_layer_sizes(state_dict)
    model = HeatPINN(
        layer_sizes,
        t_min=config.domain.t_min,
        t_max=config.domain.t_max,
        x_min=config.domain.x_min,
        x_max=config.domain.x_max,
        y_min=config.domain.y_min,
        y_max=config.domain.y_max,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, layer_sizes


def count_parameters(model: torch.nn.Module) -> int:
    """Μετρά όλες τις παραμέτρους του μοντέλου."""

    return sum(parameter.numel() for parameter in model.parameters())


def format_layer_sizes(layer_sizes: tuple[int, ...] | list[int]) -> str:
    """Γυρίζει την αρχιτεκτονική σε string."""

    return "-".join(str(size) for size in layer_sizes)


def neuron_keep_indices(model: HeatPINN, amount: float) -> list[torch.Tensor]:
    """Διαλέγει ποιους hidden neurons κρατάμε σε κάθε hidden layer."""

    layers = list(model.layers)
    keep_indices: list[torch.Tensor] = []

    for layer_index, layer in enumerate(layers[:-1]):
        next_layer = layers[layer_index + 1]

        # Score neuron = incoming weights + bias + outgoing weights.
        incoming_score = layer.weight.detach().abs().sum(dim=1).cpu()
        bias_score = layer.bias.detach().abs().cpu()
        outgoing_score = next_layer.weight.detach().abs().sum(dim=0).cpu()
        neuron_score = incoming_score + bias_score + outgoing_score

        width = neuron_score.numel()
        prune_count = min(width - 1, round(width * amount))
        keep_count = width - prune_count
        keep = torch.topk(neuron_score, k=keep_count).indices.sort().values
        keep_indices.append(keep)

    return keep_indices


def build_pruned_model(
    dense_model: HeatPINN,
    amount: float,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[HeatPINN, tuple[int, ...]]:
    """Χτίζει το νεο prunned μοντέλο κόβοντας ολόκληρα hidden neurons."""

    old_layers = list(dense_model.layers)
    keep_indices = neuron_keep_indices(dense_model, amount)

    compact_layer_sizes = [old_layers[0].in_features]
    compact_layer_sizes.extend(len(keep) for keep in keep_indices)
    compact_layer_sizes.append(old_layers[-1].out_features)

    compact_model = HeatPINN(
        tuple(compact_layer_sizes),
        t_min=config.domain.t_min,
        t_max=config.domain.t_max,
        x_min=config.domain.x_min,
        x_max=config.domain.x_max,
        y_min=config.domain.y_min,
        y_max=config.domain.y_max,
    ).to(device)

    compact_state = compact_model.state_dict()
    previous_keep: torch.Tensor | None = None

    for layer_index, old_layer in enumerate(old_layers):
        weight = old_layer.weight.detach().cpu()
        bias = old_layer.bias.detach().cpu()

        if layer_index < len(keep_indices):
            output_keep = keep_indices[layer_index]
        else:
            output_keep = torch.arange(weight.shape[0])

        weight = weight.index_select(0, output_keep)
        bias = bias.index_select(0, output_keep)

        if previous_keep is not None:
            weight = weight.index_select(1, previous_keep)

        compact_state[f"layers.{layer_index}.weight"] = weight
        compact_state[f"layers.{layer_index}.bias"] = bias

        if layer_index < len(keep_indices):
            previous_keep = output_keep

    compact_model.load_state_dict(compact_state)
    return compact_model, tuple(compact_layer_sizes)


def evaluate_model(
    model: HeatPINN,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Υπολογίζει snapshot και global metrics (όπως στα κανονικά runs)."""

    snapshot_df = evaluate_model_on_snapshots(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
    )
    summary = summarize_snapshot_metrics(snapshot_df)
    summary.update(
        evaluate_global_relative_l2(
            model,
            device,
            config.domain,
            n_space=config.evaluation.global_space_points,
            n_times=config.evaluation.global_time_points,
        )
    )
    return snapshot_df, summary


def save_pruned_checkpoint(
    checkpoint_path: Path,
    model: HeatPINN,
    layer_sizes: tuple[int, ...],
    source_checkpoint: Path,
    amount: float,
    metrics: dict[str, float],
) -> None:
    """Αποθηκεύει checkpoint και βασικά pruning metadata."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "layer_sizes": layer_sizes,
            "source_checkpoint": str(source_checkpoint),
            "pruning_algorithm": "magnitude_structured_neuron_pruning",
            "pruning_amount": amount,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def plot_accuracy(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Σχεδιάζει accuracy vs parameter reduction."""

    plot_df = summary_df.sort_values("parameter_reduction_percent")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        plot_df["parameter_reduction_percent"],
        plot_df["global_mean_relative_l2"],
        marker="o",
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            row["model_label"],
            (row["parameter_reduction_percent"], row["global_mean_relative_l2"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Parameter reduction (%)")
    ax.set_ylabel("Global mean relative L2")
    ax.set_title("Structured pruning: accuracy vs parameter reduction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Τρέχει το pruning experiment."""


    config = ExperimentConfig()
    set_seed(config.seed)
    device, device_name = get_device(config.prefer_directml)
    ensure_output_dirs(config.paths)

    best_row = load_best_baseline_row()
    baseline_run_name = str(best_row["run_name"])
    checkpoint_path = checkpoint_path_from_row(best_row)
    dense_model, dense_layer_sizes = load_dense_model(checkpoint_path, config, device)
    dense_params = count_parameters(dense_model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{baseline_run_name}_structured_pruning_{timestamp}"
    output_dir = OUTPUT_ROOT / run_name
    reports_dir = output_dir / "reports"
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    for directory in (reports_dir, models_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Device selected: {device} ({device_name})")
    print(f"Baseline run: {baseline_run_name}")
    print(f"Baseline checkpoint: {checkpoint_path}")
    print(f"Dense layer sizes: {format_layer_sizes(dense_layer_sizes)}")
    print(f"Dense parameter count: {dense_params}")
    print(f"Pruning amounts: {', '.join(f'{amount * 100:g}%' for amount in PRUNING_AMOUNTS)}")
    print(f"Output folder: {output_dir}")

    data = generate_training_data(config.domain, config.sampling, device, seed=config.seed)
    training_config = replace(
        config.training,
        lambda_f=float(best_row["lambda_f"]),
        lambda_u=float(best_row["lambda_u"]),
        lbfgs_max_iter=LBFGS_MAX_ITER,
        lbfgs_max_eval=LBFGS_MAX_EVAL,
    )

    dense_snapshot_df, dense_metrics = evaluate_model(dense_model, config, device)
    write_dataframe_report(
        reports_dir / "baseline_snapshot_metrics.xlsx",
        "Baseline Snapshot Metrics",
        dense_snapshot_df,
    )

    rows: list[dict[str, object]] = [
        {
            "model_label": "baseline_dense",
            "pruning_amount": 0.0,
            "layer_sizes": format_layer_sizes(dense_layer_sizes),
            "parameter_count": dense_params,
            "parameter_reduction_percent": 0.0,
            "global_mean_relative_l2_change_percent": 0.0,
            "checkpoint_path": str(checkpoint_path),
            **dense_metrics,
        }
    ]

    plot_snapshot_grid(
        dense_model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
        output_path=figures_dir / "baseline_dense_snapshots.png",
    )
    plot_slice_comparison(
        dense_model,
        device,
        config.domain,
        config.evaluation.slice_times,
        fixed_y=config.evaluation.fixed_y,
        n_points=config.evaluation.slice_points,
        output_path=figures_dir / "baseline_dense_slices.png",
    )

    baseline_error = dense_metrics["global_mean_relative_l2"]

    for amount in PRUNING_AMOUNTS:
        label = f"structured_neuron_{int(amount * 100):02d}"
        print(f"Running {label}...")

        pruned_model, pruned_layer_sizes = build_pruned_model(
            dense_model,
            amount,
            config,
            device,
        )
        pruned_params = count_parameters(pruned_model)

        start_time = time.perf_counter()
        lbfgs_result = fine_tune_with_lbfgs(
            pruned_model,
            data,
            training_config,
            max_iter=LBFGS_MAX_ITER,
            max_eval=LBFGS_MAX_EVAL,
        )
        fine_tune_time_sec = time.perf_counter() - start_time

        snapshot_df, metrics = evaluate_model(pruned_model, config, device)
        reduction_percent = 100.0 * (dense_params - pruned_params) / dense_params
        error_change_percent = 100.0 * (
            metrics["global_mean_relative_l2"] - baseline_error
        ) / max(baseline_error, 1e-12)

        checkpoint_out = models_dir / f"{label}.pth"
        save_pruned_checkpoint(
            checkpoint_out,
            pruned_model,
            pruned_layer_sizes,
            checkpoint_path,
            amount,
            metrics,
        )

        write_dataframe_report(
            reports_dir / f"{label}_snapshot_metrics.xlsx",
            f"{label} Snapshot Metrics",
            snapshot_df,
        )

        rows.append(
            {
                "model_label": label,
                "pruning_amount": amount,
                "layer_sizes": format_layer_sizes(pruned_layer_sizes),
                "parameter_count": pruned_params,
                "parameter_reduction_percent": reduction_percent,
                "global_mean_relative_l2_change_percent": error_change_percent,
                "best_lbfgs_loss": lbfgs_result.best_loss,
                "best_lbfgs_step": lbfgs_result.best_step,
                "fine_tune_time_sec": fine_tune_time_sec,
                "checkpoint_path": str(checkpoint_out),
                **metrics,
            }
        )

        plot_snapshot_grid(
            pruned_model,
            device,
            config.domain,
            config.evaluation.snapshot_times,
            n_plot=config.evaluation.snapshot_grid_points,
            output_path=figures_dir / f"{label}_snapshots.png",
        )
        plot_slice_comparison(
            pruned_model,
            device,
            config.domain,
            config.evaluation.slice_times,
            fixed_y=config.evaluation.fixed_y,
            n_points=config.evaluation.slice_points,
            output_path=figures_dir / f"{label}_slices.png",
        )
        plot_training_history(
            lbfgs_result.history,
            title=f"{label} L-BFGS fine-tuning history",
            xlabel="L-BFGS evaluation",
            output_path=figures_dir / f"{label}_lbfgs_history.png",
        )

    results_df = pd.DataFrame(rows)
    summary_columns = [
        "model_label",
        "pruning_amount",
        "layer_sizes",
        "parameter_count",
        "parameter_reduction_percent",
        "global_mean_relative_l2",
        "global_worst_relative_l2",
        "global_mean_relative_l2_change_percent",
        "fine_tune_time_sec",
        "checkpoint_path",
    ]
    summary_df = results_df[summary_columns].copy()

    write_dataframe_report(reports_dir / "results.xlsx", "Structured Pruning Results", results_df)
    write_dataframe_report(reports_dir / "summary.xlsx", "Structured Pruning Summary", summary_df)
    plot_accuracy(summary_df, figures_dir / "accuracy_vs_parameter_reduction.png")

    print("Saved structured pruning results to:", reports_dir / "results.xlsx")
    print("Saved structured pruning summary to:", reports_dir / "summary.xlsx")
    print(summary_df)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        finalize_process(1)
    else:
        finalize_process(0)
