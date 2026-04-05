from __future__ import annotations

"""Grid search για τα βασικά hyperparameters"""

import argparse
import copy
import gc
import traceback
from dataclasses import replace
from itertools import product
from time import perf_counter

import pandas as pd

from heat_pinn.config import (
    ExperimentConfig,
    build_baseline_adam_checkpoint_path,
    build_baseline_final_checkpoint_path,
    build_baseline_run_name,
    derive_lbfgs_max_eval,
)
from heat_pinn.data import generate_training_data
from heat_pinn.evaluation import evaluate_global_relative_l2, summarize_snapshot_metrics
from heat_pinn.experiment_log import append_experiment_rows, build_experiment_row
from heat_pinn.model import HeatPINN
from heat_pinn.plots import (
    compute_snapshot_error_vmax,
    plot_slice_comparison,
    plot_snapshot_grid,
    plot_training_history,
    plot_training_points,
)
from heat_pinn.reporting import write_dataframe_report, write_mapping_report
from heat_pinn.runtime import ensure_output_dirs, finalize_process, get_device, set_seed
from heat_pinn.training import fine_tune_with_lbfgs, train_with_adam


# Οι στήλες του αναλυτικού report
RESULTS_COLUMNS = [
    "run_name",
    "adam_epochs",
    "lbfgs_max_iter",
    "lbfgs_max_eval",
    "lambda_f",
    "lambda_u",
    "mean_relative_l2",
    "mean_mae",
    "mean_rmse",
    "mean_max_error",
    "global_mean_relative_l2",
    "global_worst_relative_l2",
    "total_train_time_sec",
]


# Οι στήλες του συνοπτικού ranking
SUMMARY_COLUMNS = [
    "rank",
    "run_name",
    "adam_epochs",
    "lbfgs_max_iter",
    "lbfgs_max_eval",
    "lambda_f",
    "lambda_u",
    "mean_relative_l2",
    "mean_rmse",
    "global_mean_relative_l2",
    "global_worst_relative_l2",
    "total_train_time_sec",
]


def parse_args() -> argparse.Namespace:
    """Διαβάζει τα CLI arguments του grid search."""

    parser = argparse.ArgumentParser(
        description="Run the hyperparameter grid search.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First combo index to consider.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last combo index to consider.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU instead of DirectML.",
    )
    return parser.parse_args()


def iter_grid_combinations(config: ExperimentConfig) -> list[tuple[int, int, float, float]]:
    """Επιστρέφει όλους τους συνδυασμούς hyperparameters του grid search."""

    # Φτιάχνω όλους τους συνδυασμούς που θα δοκιμαστούν
    grid = config.grid_search
    return list(
        product(
            grid.adam_epochs_values,
            grid.lbfgs_max_iter_values,
            grid.lambda_f_values,
            grid.lambda_u_values,
        )
    )


def build_combo_config(
    base_config: ExperimentConfig,
    *,
    combo_index: int,
    epochs_adam: int,
    lbfgs_max_iter: int,
    lambda_f: float,
    lambda_u: float,
) -> ExperimentConfig:
    """Χτίζει config για ένα συγκεκριμένο grid-search combination."""

    # Χτίζω config μόνο για αυτό το combo
    lbfgs_max_eval = derive_lbfgs_max_eval(
        lbfgs_max_iter,
        ratio=base_config.grid_search.lbfgs_max_eval_ratio,
    )
    training_config = replace(
        base_config.training,
        epochs_adam=epochs_adam,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_max_eval=lbfgs_max_eval,
        lambda_f=lambda_f,
        lambda_u=lambda_u,
    )
    return replace(
        base_config,
        training=training_config,
        run_timestamp=f"{base_config.run_timestamp}_{combo_index:03d}",
    )


def build_model(config: ExperimentConfig, device: object) -> HeatPINN:
    # Χτίζω το ίδιο μοντέλο για κάθε run
    return HeatPINN(
        config.model.layer_sizes,
        t_min=config.domain.t_min,
        t_max=config.domain.t_max,
        x_min=config.domain.x_min,
        x_max=config.domain.x_max,
        y_min=config.domain.y_min,
        y_max=config.domain.y_max,
    ).to(device)


def build_base_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    if args.cpu:
        return replace(config, prefer_directml=False)
    return config


def validate_index_range(
    total_combinations: int,
    start_index: int,
    end_index: int | None,
) -> tuple[int, int]:
    if start_index < 1:
        raise ValueError("start_index must be at least 1.")

    final_end_index = total_combinations if end_index is None else end_index
    if final_end_index > total_combinations:
        raise ValueError(
            f"end_index must be at most {total_combinations}, got {final_end_index}.",
        )
    if final_end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index.")

    return start_index, final_end_index


def build_status_row(
    *,
    grid_run_name: str,
    combo_index: int,
    combo_config: ExperimentConfig,
    run_name: str,
) -> dict[str, object]:
    return {
        "grid_run_name": grid_run_name,
        "combo_index": combo_index,
        "run_name": run_name,
        "seed": combo_config.seed,
        "adam_epochs": combo_config.training.epochs_adam,
        "lbfgs_max_iter": combo_config.training.lbfgs_max_iter,
        "lbfgs_max_eval": combo_config.training.lbfgs_max_eval,
        "lbfgs_max_eval_ratio": combo_config.grid_search.lbfgs_max_eval_ratio,
        "lambda_f": combo_config.training.lambda_f,
        "lambda_u": combo_config.training.lambda_u,
    }

def results_dataframe(results_by_index: dict[int, dict[str, object]]) -> pd.DataFrame:
    if not results_by_index:
        return pd.DataFrame()

    return pd.DataFrame(
        [results_by_index[index] for index in sorted(results_by_index)],
    )


def build_summary_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Φτιάχνει το τελικό ranking των επιτυχημένων combinations."""

    # Το "best" combo ορίζεται από την εγγύτητα της πρόβλεψης u_pred
    # προς την exact solution του benchmark.
    # Κάνω ranking πρώτα με βάση το global_mean_relative_l2
    # και σε ισοβαθμία με βάση το global_worst_relative_l2.
    completed_df = results_df[results_df["status"] == "completed"].copy()
    if completed_df.empty:
        return pd.DataFrame()

    for column in RESULTS_COLUMNS:
        if column not in completed_df.columns:
            completed_df[column] = pd.NA

    review_results_df = completed_df[RESULTS_COLUMNS].copy()
    summary_df = review_results_df.sort_values(
        by=[
            "global_mean_relative_l2",
            "global_worst_relative_l2",
        ]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
    return summary_df[SUMMARY_COLUMNS].copy()


def persist_grid_search_reports(
    config: ExperimentConfig,
    grid_run_name: str,
    results_by_index: dict[int, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Γράφει τα τρέχοντα results, summary και best reports του grid search."""

    results_df = results_dataframe(results_by_index)
    if results_df.empty:
        return results_df, pd.DataFrame()

    write_dataframe_report(
        config.paths.grid_search_report_path(grid_run_name, "results"),
        "Baseline Grid Search Results",
        results_df,
    )

    summary_df = build_summary_dataframe(results_df)
    if not summary_df.empty:
        write_dataframe_report(
            config.paths.grid_search_report_path(grid_run_name, "summary"),
            "Baseline Grid Search Summary",
            summary_df,
        )
        write_mapping_report(
            config.paths.grid_search_report_path(grid_run_name, "best"),
            "Baseline Grid Search Best",
            summary_df.iloc[0].to_dict(),
        )

    return results_df, summary_df


def run_single_combo(
    *,
    grid_run_name: str,
    combo_index: int,
    combo_config: ExperimentConfig,
    run_name: str,
    data: object,
    device: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Τρέχει ένα combination του grid search και επιστρέφει τα reports του."""

    set_seed(combo_config.seed)
    model = build_model(combo_config, device)
    adam_checkpoint_path = build_baseline_adam_checkpoint_path(combo_config)
    final_checkpoint_path = build_baseline_final_checkpoint_path(combo_config)

    # Τα training points μένουν ίδια για δίκαιη σύγκριση
    plot_training_points(
        data,
        combo_config.domain,
        output_path=combo_config.paths.run_training_points_figure_path(run_name),
    )

    adam_start = perf_counter()
    adam_result = train_with_adam(
        model,
        data,
        combo_config.training,
        checkpoint_path=adam_checkpoint_path,
    )
    adam_time_sec = perf_counter() - adam_start
    adam_state_dict = copy.deepcopy(model.state_dict())

    # Κρατάω τις μετρικές μετά το Adam
    adam_snapshot_df = plot_snapshot_grid(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.snapshot_times,
        n_plot=combo_config.evaluation.snapshot_grid_points,
        output_path=None,
    )
    adam_summary = summarize_snapshot_metrics(adam_snapshot_df)
    adam_global = evaluate_global_relative_l2(
        model,
        device,
        combo_config.domain,
        n_space=combo_config.evaluation.global_space_points,
        n_times=combo_config.evaluation.global_time_points,
    )
    plot_training_history(
        adam_result.history,
        title="Adam training history",
        xlabel="Epoch",
        output_path=combo_config.paths.run_model_history_figure_path(run_name, "baseline_best_adam"),
    )
    plot_slice_comparison(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.slice_times,
        fixed_y=combo_config.evaluation.fixed_y,
        n_points=combo_config.evaluation.slice_points,
        output_path=combo_config.paths.run_model_slice_figure_path(run_name, "baseline_best_adam"),
    )

    lbfgs_start = perf_counter()
    lbfgs_result = fine_tune_with_lbfgs(
        model,
        data,
        combo_config.training,
        checkpoint_path=final_checkpoint_path,
    )
    lbfgs_time_sec = perf_counter() - lbfgs_start
    final_state_dict = copy.deepcopy(model.state_dict())

    # Υπολογίζω κοινό error range από Adam και final μέσα στο ίδιο combo
    model.load_state_dict(adam_state_dict)
    shared_snapshot_error_vmax = compute_snapshot_error_vmax(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.snapshot_times,
        n_plot=combo_config.evaluation.snapshot_grid_points,
    )
    model.load_state_dict(final_state_dict)
    shared_snapshot_error_vmax = max(
        shared_snapshot_error_vmax,
        compute_snapshot_error_vmax(
            model,
            device,
            combo_config.domain,
            combo_config.evaluation.snapshot_times,
            n_plot=combo_config.evaluation.snapshot_grid_points,
        ),
    )

    model.load_state_dict(adam_state_dict)
    plot_snapshot_grid(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.snapshot_times,
        n_plot=combo_config.evaluation.snapshot_grid_points,
        error_vmax=shared_snapshot_error_vmax,
        output_path=combo_config.paths.run_model_snapshot_figure_path(run_name, "baseline_best_adam"),
    )
    model.load_state_dict(final_state_dict)

    # Κρατάω τα τελικά metrics του combo
    plot_training_history(
        lbfgs_result.history,
        title="L-BFGS fine-tuning history",
        xlabel="L-BFGS iteration",
        output_path=combo_config.paths.run_model_history_figure_path(run_name, "baseline_final_lbfgs"),
    )

    snapshot_df = plot_snapshot_grid(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.snapshot_times,
        n_plot=combo_config.evaluation.snapshot_grid_points,
        error_vmax=shared_snapshot_error_vmax,
        output_path=combo_config.paths.run_model_snapshot_figure_path(run_name, "baseline_final_lbfgs"),
    )
    snapshot_summary = summarize_snapshot_metrics(snapshot_df)
    global_summary = evaluate_global_relative_l2(
        model,
        device,
        combo_config.domain,
        n_space=combo_config.evaluation.global_space_points,
        n_times=combo_config.evaluation.global_time_points,
    )
    write_dataframe_report(
        combo_config.paths.run_result_path(run_name, "baseline_snapshot_metrics"),
        "Baseline Snapshot Metrics",
        snapshot_df,
    )
    write_mapping_report(
        combo_config.paths.run_result_path(run_name, "baseline_global_summary"),
        "Baseline Global Summary",
        {**snapshot_summary, **global_summary},
    )
    plot_slice_comparison(
        model,
        device,
        combo_config.domain,
        combo_config.evaluation.slice_times,
        fixed_y=combo_config.evaluation.fixed_y,
        n_points=combo_config.evaluation.slice_points,
        output_path=combo_config.paths.run_model_slice_figure_path(run_name, "baseline_final_lbfgs"),
    )

    row: dict[str, object] = {
        "grid_run_name": grid_run_name,
        "combo_index": combo_index,
        "run_name": run_name,
        "status": "completed",
        "seed": combo_config.seed,
        "adam_epochs": combo_config.training.epochs_adam,
        "lbfgs_max_iter": combo_config.training.lbfgs_max_iter,
        "lbfgs_max_eval": combo_config.training.lbfgs_max_eval,
        "lbfgs_max_eval_ratio": combo_config.grid_search.lbfgs_max_eval_ratio,
        "lambda_f": combo_config.training.lambda_f,
        "lambda_u": combo_config.training.lambda_u,
        "best_adam_loss": adam_result.best_loss,
        "best_adam_epoch": adam_result.best_epoch,
        "adam_mean_relative_l2": adam_global["global_mean_relative_l2"],
        "adam_worst_relative_l2": adam_global["global_worst_relative_l2"],
        "best_lbfgs_loss": lbfgs_result.best_loss,
        "best_lbfgs_step": lbfgs_result.best_step,
        "adam_time_sec": adam_time_sec,
        "lbfgs_time_sec": lbfgs_time_sec,
        "total_train_time_sec": adam_time_sec + lbfgs_time_sec,
        "adam_checkpoint_path": str(adam_checkpoint_path),
        "final_checkpoint_path": str(final_checkpoint_path),
    }
    row.update(snapshot_summary)
    row.update(global_summary)

    experiment_row = build_experiment_row(
        config=combo_config,
        model_label="grid_search_final",
        metrics={**snapshot_summary, **global_summary},
        extra={
            "grid_run_name": grid_run_name,
            "combo_index": combo_index,
            "run_name": run_name,
            "adam_snapshot_mean_relative_l2": adam_summary["mean_relative_l2"],
            "adam_global_mean_relative_l2": adam_global["global_mean_relative_l2"],
            "best_adam_loss": adam_result.best_loss,
            "best_lbfgs_loss": lbfgs_result.best_loss,
            "adam_time_sec": adam_time_sec,
            "lbfgs_time_sec": lbfgs_time_sec,
            "total_train_time_sec": adam_time_sec + lbfgs_time_sec,
            "status": "completed",
            "pruning_type": "none",
            "pruning_amount": 0.0,
        },
    )

    return row, experiment_row


def build_failure_row(
    *,
    grid_run_name: str,
    combo_index: int,
    combo_config: ExperimentConfig,
    run_name: str,
    error: Exception,
) -> dict[str, object]:
    row = build_status_row(
        grid_run_name=grid_run_name,
        combo_index=combo_index,
        combo_config=combo_config,
        run_name=run_name,
    )
    row.update(
        {
            "status": "failed",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )
    return row

def print_failed_rows(results_df: pd.DataFrame) -> None:
    failed_df = results_df[results_df["status"] != "completed"].copy()
    if failed_df.empty:
        return

    print(f"Non-completed combinations: {len(failed_df)}")
    columns = [
        "combo_index",
        "run_name",
        "status",
        "adam_epochs",
        "lbfgs_max_iter",
        "lbfgs_max_eval",
        "lambda_f",
        "lambda_u",
        "error",
    ]
    available_columns = [column for column in columns if column in failed_df.columns]
    print(failed_df[available_columns].to_string(index=False))


def format_optional_seconds(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


def main() -> None:
    """Τρέχει το full grid-search pipeline και γράφει τα συγκεντρωτικά reports."""

    args = parse_args()

    # Στήνω το grid search run
    config = build_base_config(args)
    combinations = iter_grid_combinations(config)
    start_index, end_index = validate_index_range(
        len(combinations),
        args.start_index,
        args.end_index,
    )
    grid_run_name = f"baseline_grid_search_{config.run_timestamp}"

    set_seed(config.seed)
    device, device_name = get_device(config.prefer_directml)
    ensure_output_dirs(config.paths)
    data = generate_training_data(config.domain, config.sampling, device, seed=config.seed)

    results_by_index: dict[int, dict[str, object]] = {}

    print(f"Device selected: {device} ({device_name})")
    print(f"Grid run name: {grid_run_name}")
    print(f"Total combinations: {len(combinations)}")
    print(f"Selected index range: {start_index}..{end_index}")
    print(
        "L-BFGS policy: "
        f"max_eval ~= {config.grid_search.lbfgs_max_eval_ratio:g} * max_iter"
    )
    print(
        "Ranking policy: lowest global_mean_relative_l2 "
        "(u_pred vs exact solution), tie-break with global_worst_relative_l2"
    )

    experiment_rows: list[dict[str, object]] = []

    # Τρέχω κάθε συνδυασμό ξεχωριστά
    for combo_index, (epochs_adam, lbfgs_max_iter, lambda_f, lambda_u) in enumerate(combinations, start=1):
        if combo_index < start_index or combo_index > end_index:
            continue

        combo_config = build_combo_config(
            config,
            combo_index=combo_index,
            epochs_adam=epochs_adam,
            lbfgs_max_iter=lbfgs_max_iter,
            lambda_f=lambda_f,
            lambda_u=lambda_u,
        )
        run_name = build_baseline_run_name(combo_config)

        print(
            f"[{combo_index}/{len(combinations)}] "
            f"epochs={epochs_adam}, "
            f"lbfgs_iter={lbfgs_max_iter}, "
            f"lbfgs_eval={combo_config.training.lbfgs_max_eval}, "
            f"lambda_f={lambda_f:g}, "
            f"lambda_u={lambda_u:g}"
        )

        try:
            row, experiment_row = run_single_combo(
                grid_run_name=grid_run_name,
                combo_index=combo_index,
                combo_config=combo_config,
                run_name=run_name,
                data=data,
                device=device,
            )
        except Exception as exc:
            results_by_index[combo_index] = build_failure_row(
                grid_run_name=grid_run_name,
                combo_index=combo_index,
                combo_config=combo_config,
                run_name=run_name,
                error=exc,
            )
            persist_grid_search_reports(config, grid_run_name, results_by_index)
            print(f"Combination {combo_index} failed: {exc}")
            print(traceback.format_exc())
        else:
            results_by_index[combo_index] = row
            experiment_rows.append(experiment_row)
            persist_grid_search_reports(config, grid_run_name, results_by_index)
        finally:
            gc.collect()

    results_df, summary_df = persist_grid_search_reports(config, grid_run_name, results_by_index)
    if results_df.empty:
        raise RuntimeError("Grid search finished with zero recorded runs.")

    print_failed_rows(results_df)

    if summary_df.empty:
        raise RuntimeError("Grid search finished with zero successful runs.")

    if experiment_rows:
        append_experiment_rows(config.paths.experiment_registry_path, experiment_rows)

    best_run = summary_df.iloc[0]

    # Print το καλύτερο run στο τέλος
    print("Saved grid-search results to:", config.paths.grid_search_report_path(grid_run_name, "results"))
    print("Saved grid-search summary to:", config.paths.grid_search_report_path(grid_run_name, "summary"))
    print(
        "WINNER: "
        f"run_name={best_run['run_name']}, "
        f"adam_epochs={int(best_run['adam_epochs'])}, "
        f"lbfgs_max_iter={int(best_run['lbfgs_max_iter'])}, "
        f"lbfgs_max_eval={int(best_run['lbfgs_max_eval'])}, "
        f"lambda_f={float(best_run['lambda_f']):g}, "
        f"lambda_u={float(best_run['lambda_u']):g}, "
        f"global_mean_relative_l2={float(best_run['global_mean_relative_l2']):.6e}, "
        f"global_worst_relative_l2={float(best_run['global_worst_relative_l2']):.6e}, "
        f"total_train_time_sec={format_optional_seconds(best_run['total_train_time_sec'])}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        finalize_process(1)
    else:
        finalize_process(0)
