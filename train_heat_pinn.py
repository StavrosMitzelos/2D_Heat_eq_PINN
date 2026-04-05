from __future__ import annotations

"""Εκπαίδευση του βασικού PINN"""

import copy
import traceback

import pandas as pd

from heat_pinn.config import (
    ExperimentConfig,
    build_baseline_adam_checkpoint_path,
    build_baseline_final_checkpoint_path,
    build_baseline_run_name,
)
from heat_pinn.data import generate_training_data
from heat_pinn.evaluation import (
    evaluate_global_relative_l2,
    summarize_snapshot_metrics,
)
from heat_pinn.experiment_log import append_experiment_rows, build_experiment_row
from heat_pinn.model import HeatPINN
from heat_pinn.plots import (
    compute_snapshot_error_vmax,
    compute_spacetime_error_vmax,
    plot_slice_comparison,
    plot_spacetime_slice_grid,
    plot_snapshot_grid,
    plot_training_history,
    plot_training_points,
)
from heat_pinn.reporting import write_dataframe_report, write_mapping_report
from heat_pinn.runtime import ensure_output_dirs, finalize_process, get_device, set_seed
from heat_pinn.training import fine_tune_with_lbfgs, train_with_adam


def main() -> None:
    """Τρέχει το baseline pipeline: training, evaluation, reports και plots."""

    # Στήνω το βασικό run
    config = ExperimentConfig()
    adam_label = "baseline_best_adam"
    final_label = "baseline_final_lbfgs"
    run_name = build_baseline_run_name(config)
    set_seed(config.seed)
    device, device_name = get_device(config.prefer_directml)
    ensure_output_dirs(config.paths)
    named_adam_path = build_baseline_adam_checkpoint_path(config)
    named_final_path = build_baseline_final_checkpoint_path(config)

    print(f"Device selected: {device} ({device_name})")
    print(f"Run name: {run_name}")

    # Φτιάχνω τα training points
    data = generate_training_data(config.domain, config.sampling, device, seed=config.seed)
    print(f"Collocation points: {data.t_f.shape[0]}")
    print(f"Supervised points: {data.t_u.shape[0]}")

    plot_training_points(
        data,
        config.domain,
        output_path=config.paths.run_training_points_figure_path(run_name),
    )

    # Χτίζω το PINN
    model = HeatPINN(
        config.model.layer_sizes,
        t_min=config.domain.t_min,
        t_max=config.domain.t_max,
        x_min=config.domain.x_min,
        x_max=config.domain.x_max,
        y_min=config.domain.y_min,
        y_max=config.domain.y_max,
    ).to(device)
    print(model)

    # Πρώτα κάνω την εκπαίδευση με Adam
    adam_result = train_with_adam(
        model,
        data,
        config.training,
        checkpoint_path=named_adam_path,
    )
    print(f"Best Adam loss: {adam_result.best_loss:.6e} at epoch {adam_result.best_epoch}")
    print(f"Saved Adam checkpoint to: {named_adam_path}")
    adam_state_dict = copy.deepcopy(model.state_dict())

    # Μετράω το μοντέλο μετά το Adam
    adam_snapshot_df = plot_snapshot_grid(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
        output_path=config.paths.run_model_snapshot_figure_path(run_name, adam_label),
    )
    adam_summary = summarize_snapshot_metrics(adam_snapshot_df)
    adam_global = evaluate_global_relative_l2(
        model,
        device,
        config.domain,
        n_space=config.evaluation.global_space_points,
        n_times=config.evaluation.global_time_points,
    )

    plot_training_history(
        adam_result.history,
        title="Adam training history",
        xlabel="Epoch",
        output_path=config.paths.run_model_history_figure_path(run_name, adam_label),
    )
    plot_spacetime_slice_grid(
        model,
        device,
        config.domain,
        fixed_y=config.evaluation.fixed_y,
        snapshot_times=config.evaluation.snapshot_times,
        output_path=config.paths.run_model_spacetime_figure_path(run_name, adam_label),
    )
    plot_slice_comparison(
        model,
        device,
        config.domain,
        config.evaluation.slice_times,
        fixed_y=config.evaluation.fixed_y,
        n_points=config.evaluation.slice_points,
        output_path=config.paths.run_model_slice_figure_path(run_name, adam_label),
    )

    append_experiment_rows(
        config.paths.experiment_registry_path,
        [
            build_experiment_row(
                config=config,
                model_label=adam_label,
                metrics={**adam_summary, **adam_global},
                extra={
                    "run_name": run_name,
                    "best_adam_loss": adam_result.best_loss,
                    "pruning_type": "none",
                    "pruning_amount": 0.0,
                },
            )
        ],
    )
    print("Appended Adam experiment row to:", config.paths.experiment_registry_path)

    print("Starting L-BFGS fine-tuning...")

    # Μετά κάνω fine-tuning με L-BFGS
    lbfgs_result = fine_tune_with_lbfgs(
        model,
        data,
        config.training,
        checkpoint_path=named_final_path,
    )
    print(
        f"Best L-BFGS loss: {lbfgs_result.best_loss:.6e} "
        f"at evaluation {lbfgs_result.best_step}"
    )
    print(f"Saved final checkpoint to: {named_final_path}")

    final_state_dict = copy.deepcopy(model.state_dict())

    # Υπολογίζω κοινό error range από Adam και final για σύγκριση στα plots
    model.load_state_dict(adam_state_dict)
    shared_snapshot_error_vmax = compute_snapshot_error_vmax(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
    )
    shared_spacetime_error_vmax = compute_spacetime_error_vmax(
        model,
        device,
        config.domain,
        fixed_y=config.evaluation.fixed_y,
    )

    model.load_state_dict(final_state_dict)
    shared_snapshot_error_vmax = max(
        shared_snapshot_error_vmax,
        compute_snapshot_error_vmax(
            model,
            device,
            config.domain,
            config.evaluation.snapshot_times,
            n_plot=config.evaluation.snapshot_grid_points,
        ),
    )
    shared_spacetime_error_vmax = max(
        shared_spacetime_error_vmax,
        compute_spacetime_error_vmax(
            model,
            device,
            config.domain,
            fixed_y=config.evaluation.fixed_y,
        ),
    )

    model.load_state_dict(adam_state_dict)
    plot_snapshot_grid(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
        error_vmax=shared_snapshot_error_vmax,
        output_path=config.paths.run_model_snapshot_figure_path(run_name, adam_label),
    )
    plot_spacetime_slice_grid(
        model,
        device,
        config.domain,
        fixed_y=config.evaluation.fixed_y,
        snapshot_times=config.evaluation.snapshot_times,
        error_vmax=shared_spacetime_error_vmax,
        output_path=config.paths.run_model_spacetime_figure_path(run_name, adam_label),
    )

    model.load_state_dict(final_state_dict)

    plot_training_history(
        lbfgs_result.history,
        title="L-BFGS fine-tuning history",
        xlabel="L-BFGS iteration",
        output_path=config.paths.run_model_history_figure_path(run_name, final_label),
    )
    plot_spacetime_slice_grid(
        model,
        device,
        config.domain,
        fixed_y=config.evaluation.fixed_y,
        snapshot_times=config.evaluation.snapshot_times,
        error_vmax=shared_spacetime_error_vmax,
        output_path=config.paths.run_model_spacetime_figure_path(run_name, final_label),
    )

    # Κρατάω τα τελικά metrics και plots
    snapshot_df = plot_snapshot_grid(
        model,
        device,
        config.domain,
        config.evaluation.snapshot_times,
        n_plot=config.evaluation.snapshot_grid_points,
        error_vmax=shared_snapshot_error_vmax,
        output_path=config.paths.run_model_snapshot_figure_path(run_name, final_label),
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

    plot_slice_comparison(
        model,
        device,
        config.domain,
        config.evaluation.slice_times,
        fixed_y=config.evaluation.fixed_y,
        n_points=config.evaluation.slice_points,
        output_path=config.paths.run_model_slice_figure_path(run_name, final_label),
    )

    print("Average baseline-model metrics across snapshots:")
    print(snapshot_df[["relative_l2", "mae", "rmse", "max_error"]].mean())
    print("Global baseline-model metrics:")
    print(pd.DataFrame([baseline_global]))

    # Γράφω το τελικό row στο registry
    experiment_rows = [
        build_experiment_row(
            config=config,
            model_label=final_label,
            metrics={**baseline_summary, **baseline_global},
            extra={
                "run_name": run_name,
                "best_adam_loss": adam_result.best_loss,
                "best_lbfgs_loss": lbfgs_result.best_loss,
                "pruning_type": "none",
                "pruning_amount": 0.0,
            },
        ),
    ]
    append_experiment_rows(config.paths.experiment_registry_path, experiment_rows)
    print("Appended L-BFGS experiment row to:", config.paths.experiment_registry_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        finalize_process(1)
    else:
        finalize_process(0)
