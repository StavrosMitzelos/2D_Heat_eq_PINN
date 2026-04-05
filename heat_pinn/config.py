from __future__ import annotations


from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DomainConfig:
    """Όρια του χωροχρονικού πεδίου"""

    t_min: float = 0.0
    t_max: float = 1.0
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0


@dataclass(frozen=True)
class SamplingConfig:
    """Πλήθος collocation σημείων, αρχική και συνοριακή συνθήκη"""

    n_f: int = 10_000
    n_i: int = 1_000
    n_b: int = 500


@dataclass(frozen=True)
class ModelConfig:
    """Αρχιτεκτονική του νευρωνικού δικτύου"""

    layer_sizes: tuple[int, ...] = (3, 20, 20, 20, 20, 20, 20, 20, 20, 1)


@dataclass(frozen=True)
class TrainingConfig:
    """Παράμετροι εκπαίδευσης για Adam και L-BFGS"""

    epochs_adam: int = 500
    adam_lr: float = 1e-3
    lambda_f: float = 1.0
    lambda_u: float = 100.0
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 250
    lbfgs_max_eval: int = 312
    lbfgs_history_size: int = 50
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 2.220446049250313e-16


@dataclass(frozen=True)
class EvaluationConfig:
    """Ρυθμίσεις για snapshots, slices και συνολική αξιολόγηση"""

    snapshot_times: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    slice_times: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    fixed_y: float = 0.5
    snapshot_grid_points: int = 100
    slice_points: int = 200
    global_space_points: int = 80
    global_time_points: int = 11


@dataclass(frozen=True)
class PruningConfig:
    """Ρυθμίσεις για τα pruning πειράματα"""

    unstructured_amounts: tuple[float, ...] = (0.10, 0.20, 0.40)
    structured_amounts: tuple[float, ...] = (0.10, 0.15, 0.20)
    lbfgs_max_iter: int = 1_000


@dataclass(frozen=True)
class GridSearchConfig:
    """Τιμές υπερπαραμέτρων για το grid search"""

    adam_epochs_values: tuple[int, ...] = (500, 2_000, 5_000)
    lbfgs_max_iter_values: tuple[int, ...] = (250, 500)
    lambda_f_values: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    lambda_u_values: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0, 1_000.0)
    lbfgs_max_eval_ratio: float = 1.25


@dataclass(frozen=True)
class PathsConfig:
    """Όλα τα paths εξόδου του project"""

    results_dir: Path = field(default_factory=lambda: Path("results"))
    runs_dir: Path = field(default_factory=lambda: Path("runs"))
    pruning_sweeps_dir: Path = field(default_factory=lambda: Path("pruning_sweeps"))

    # Φτιάχνει τον βασικό φάκελο ενός run
    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    # Φάκελος checkpoints ενός run
    def run_models_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "models"

    # Φάκελος reports ενός run
    def run_reports_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "reports"

    # Φάκελος figures ενός run
    def run_figure_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "figures"

    def run_model_checkpoint_path(self, run_name: str, model_name: str) -> Path:
        return self.run_models_dir(run_name) / f"{model_name}.pth"

    def run_result_path(self, run_name: str, report_name: str) -> Path:
        return self.run_reports_dir(run_name) / f"{report_name}.xlsx"

    def run_training_points_figure_path(self, run_name: str) -> Path:
        return self.run_figure_dir(run_name) / "training_points.png"

    def run_model_snapshot_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.run_figure_dir(run_name) / f"{model_label}_snapshots.png"

    def run_model_slice_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.run_figure_dir(run_name) / f"{model_label}_slices.png"

    def run_model_history_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.run_figure_dir(run_name) / f"{model_label}_history.png"

    def run_model_spacetime_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.run_figure_dir(run_name) / f"{model_label}_spacetime_slice.png"

    def pruning_run_model_snapshot_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.pruning_run_figures_dir(run_name) / f"{model_label}_snapshots.png"

    def pruning_run_model_slice_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.pruning_run_figures_dir(run_name) / f"{model_label}_slices.png"

    def pruning_run_model_history_figure_path(self, run_name: str, model_label: str) -> Path:
        return self.pruning_run_figures_dir(run_name) / f"{model_label}_history.png"

    def run_pruning_sparsity_figure_path(self, run_name: str) -> Path:
        return self.pruning_run_figures_dir(run_name) / "pruning_sweep_accuracy_vs_sparsity.png"

    def run_pruning_nonzero_figure_path(self, run_name: str) -> Path:
        return self.pruning_run_figures_dir(run_name) / "pruning_sweep_accuracy_vs_nonzero.png"

    def grid_search_dir(self, grid_run_name: str) -> Path:
        return self.results_dir / "grid_searches" / grid_run_name

    def grid_search_report_path(self, grid_run_name: str, report_name: str) -> Path:
        return self.grid_search_dir(grid_run_name) / f"{report_name}.xlsx"

    def pruning_run_dir(self, run_name: str) -> Path:
        return self.pruning_sweeps_dir / run_name

    def pruning_run_models_dir(self, run_name: str) -> Path:
        return self.pruning_run_dir(run_name) / "models"

    def pruning_run_reports_dir(self, run_name: str) -> Path:
        return self.pruning_run_dir(run_name) / "reports"

    def pruning_run_figures_dir(self, run_name: str) -> Path:
        return self.pruning_run_dir(run_name) / "figures"

    def pruning_run_report_path(self, run_name: str, report_name: str) -> Path:
        return self.pruning_run_reports_dir(run_name) / f"{report_name}.xlsx"

    def pruning_run_model_checkpoint_path(
        self,
        run_name: str,
        model_name: str,
    ) -> Path:
        return self.pruning_run_models_dir(run_name) / f"{model_name}.pth"

    @property
    def experiment_registry_path(self) -> Path:
        return self.runs_dir / "experiment_registry.xlsx"

    @property
    def legacy_training_points_figure_path(self) -> Path:
        return Path("figures") / "training_points.png"

    @property
    def legacy_adam_history_figure_path(self) -> Path:
        return Path("figures") / "baseline_best_adam_history.png"

    @property
    def legacy_lbfgs_history_figure_path(self) -> Path:
        return Path("figures") / "baseline_final_lbfgs_history.png"

    @property
    def legacy_snapshot_figure_path(self) -> Path:
        return Path("figures") / "baseline_final_lbfgs_snapshots.png"

    @property
    def legacy_slice_figure_path(self) -> Path:
        return Path("figures") / "baseline_final_lbfgs_slices.png"

    @property
    def legacy_pruning_sparsity_figure_path(self) -> Path:
        return Path("figures") / "pruning_sweep_accuracy_vs_sparsity.png"

    @property
    def legacy_pruning_nonzero_figure_path(self) -> Path:
        return Path("figures") / "pruning_sweep_accuracy_vs_nonzero.png"


@dataclass(frozen=True)
class ExperimentConfig:
    """Συγκεντρώνει όλες τις επιμέρους ρυθμίσεις σε ένα object"""

    seed: int = 42
    prefer_directml: bool = True
    run_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    domain: DomainConfig = field(default_factory=DomainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def derive_lbfgs_max_eval(max_iter: int, ratio: float = 1.25) -> int:
    """Υπολογίζει το max_eval ως αναλογία του max_iter"""

    return max(1, int(ratio * max_iter))


def build_baseline_run_stem(config: ExperimentConfig) -> str:
    """Χτίζει το βασικό όνομα ενός baseline run χωρίς timestamp"""

    return (
        f"baseline_{config.training.epochs_adam}_"
        f"{config.training.lbfgs_max_iter}-{config.training.lbfgs_max_eval}_"
        f"{config.training.lambda_f:g}.{config.training.lambda_u:g}"
    )


def build_baseline_run_name(config: ExperimentConfig) -> str:
    """Χτίζει το πλήρες όνομα ενός baseline run"""

    return f"{build_baseline_run_stem(config)}_{config.run_timestamp}"


def build_baseline_adam_checkpoint_path(config: ExperimentConfig) -> Path:
    """Path για το καλύτερο Adam checkpoint"""

    return config.paths.run_model_checkpoint_path(build_baseline_run_name(config), "baseline_best_adam")


def build_baseline_final_checkpoint_path(config: ExperimentConfig) -> Path:
    """Path για το τελικό checkpoint μετά το L-BFGS"""

    return config.paths.run_model_checkpoint_path(build_baseline_run_name(config), "baseline_final_lbfgs")


def build_evaluation_run_name(baseline_run_name: str, config: ExperimentConfig) -> str:
    """Όνομα run για το script αξιολόγησης"""

    return f"{baseline_run_name}_evaluation_{config.run_timestamp}"


def build_pruning_run_name(baseline_run_name: str, config: ExperimentConfig) -> str:
    """Όνομα run για το pruning sweep"""

    return f"{baseline_run_name}_pruning_{config.run_timestamp}"


def build_pruned_checkpoint_path(
    paths: PathsConfig,
    pruning_run_name: str,
    pruning_type: str,
    pruning_amount: float,
) -> Path:
    """Path για checkpoint pruned μοντέλου"""

    return paths.pruning_run_model_checkpoint_path(
        pruning_run_name,
        f"{pruning_type}_{int(pruning_amount * 100):02d}",
    )
