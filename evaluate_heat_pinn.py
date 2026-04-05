from __future__ import annotations

"""Αξιολόγηση ενός ήδη εκπαιδευμένου PINN"""

import traceback

import pandas as pd

from heat_pinn.config import (
    ExperimentConfig,
    build_evaluation_run_name,
)
from heat_pinn.evaluation import evaluate_global_relative_l2, summarize_snapshot_metrics
from heat_pinn.model import HeatPINN
from heat_pinn.plots import plot_slice_comparison, plot_snapshot_grid, plot_spacetime_slice_grid
from heat_pinn.reporting import write_dataframe_report, write_mapping_report
from heat_pinn.runtime import (
    baseline_run_name_from_checkpoint_path,
    ensure_checkpoint_exists,
    ensure_output_dirs,
    finalize_process,
    get_device,
    load_checkpoint,
    resolve_latest_baseline_checkpoint_path,
    set_seed,
)


def main() -> None:
    """Φορτώνει το πιο πρόσφατο συμβατό baseline checkpoint και το αξιολογεί."""

    # Στήνω το evaluation run
    config = ExperimentConfig()
    model_label = "baseline_final_lbfgs"
    set_seed(config.seed)
    device, device_name = get_device(config.prefer_directml)
    ensure_output_dirs(config.paths)
    checkpoint_path = resolve_latest_baseline_checkpoint_path(config, checkpoint_kind="final")
    baseline_run_name = baseline_run_name_from_checkpoint_path(checkpoint_path)
    run_name = build_evaluation_run_name(baseline_run_name, config)

    print(f"Device selected: {device} ({device_name})")
    print(f"Run name: {run_name}")
    print(f"Baseline checkpoint run: {baseline_run_name}")

    # Χτίζω το ίδιο μοντέλο και φορτώνω τα weights
    model = HeatPINN(
        config.model.layer_sizes,
        t_min=config.domain.t_min,
        t_max=config.domain.t_max,
        x_min=config.domain.x_min,
        x_max=config.domain.x_max,
        y_min=config.domain.y_min,
        y_max=config.domain.y_max,
    ).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    ensure_checkpoint_exists(checkpoint_path)
    state_dict = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    # Βγάζω τα snapshot metrics και τα plots
    snapshot_df = plot_snapshot_grid(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
        output_path=config.paths.run_model_snapshot_figure_path(run_name, model_label),
    )
    write_dataframe_report(
        config.paths.run_result_path(run_name, "baseline_snapshot_metrics"),
        "Baseline Snapshot Metrics",
        snapshot_df,
    )

    baseline_summary = summarize_snapshot_metrics(snapshot_df)
    baseline_global = evaluate_global_relative_l2(
        model,
        device,
        config.domain,
        n_space=config.evaluation.global_space_points,
        n_times=config.evaluation.global_time_points,
    )
    write_mapping_report(
        config.paths.run_result_path(run_name, "baseline_global_summary"),
        "Baseline Global Summary",
        {**baseline_summary, **baseline_global},
    )

    plot_spacetime_slice_grid(
        model,
        device,
        config.domain,
        fixed_y=config.evaluation.fixed_y,
        snapshot_times=config.evaluation.snapshot_times,
        output_path=config.paths.run_model_spacetime_figure_path(run_name, model_label),
    )

    # Κρατάω και τις 1D τομές για σύγκριση
    plot_slice_comparison(
        model,
        device,
        config.domain,
        config.evaluation.slice_times,
        fixed_y=config.evaluation.fixed_y,
        n_points=config.evaluation.slice_points,
        output_path=config.paths.run_model_slice_figure_path(run_name, model_label),
    )

    print("Average baseline-model metrics across snapshots:")
    print(snapshot_df[["relative_l2", "mae", "rmse", "max_error"]].mean())
    print("Global baseline-model metrics:")
    print(pd.DataFrame([baseline_global]))


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(exc)
        finalize_process(1)
    except Exception:
        traceback.print_exc()
        finalize_process(1)
    else:
        finalize_process(0)
