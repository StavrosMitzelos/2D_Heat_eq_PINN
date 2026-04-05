# PINN για τη 2D Εξίσωση Θερμότητας

- `heat_eq.ipynb`: το αρχικό notebook, όπου έστησα πρώτα το PINN και έκανα τα πρώτα πειράματα
- `heat_pinn/`: τα scripts του project σε πιο οργανωμένη μορφή
- `run_hyperparameter_grid.py`: grid search για να βρω τα καλύτερα hyperparameters

έχω κάνει και κάποια pruning experiments (structured, unstructured) πιο πολύ για να είναι έτοιμος ο κώδικας όταν κοιτάξουμε το pruning με NAS.


## Files

- `train_heat_pinn.py`
  Το βασικό script του baseline. Τρέχει Adam, μετά L-BFGS, και μέσα στο ίδιο script κάνει και την αξιολόγηση του Adam checkpoint και του τελικού checkpoint (L-BFGS).

- `evaluate_heat_pinn.py`
  Script μόνο για evaluation. Φορτώνει ένα ήδη εκπαιδευμένο baseline checkpoint και ξαναγράφει reports και plots. (Το `train_heat_pinn.py` πραγματοποιεί ήδη evaluation οπότε το `evaluate_heat_pinn.py` είναι για την περίπτωση που θέλω να κάνω μόνο evaluation)

- `run_hyperparameter_grid.py`
  Τρέχει grid search στα βασικά hyperparameters του baseline pipeline.

- `heat_pinn/`
  Το package με όλη τη λογική του project.

- `results/`
  Τα συγκεντρωτικά αποτελέσματα του grid search.

- `example_runs/`
  Μερικά αντιπροσωπευτικά runs που κράτησα σαν δείγμα, μαζί με reports / plots / checkpoints.

## Πώς είναι οργανωμένο το `heat_pinn`

- `config.py`
  Όλες οι ρυθμίσεις του project: domain, sampling, model, training, evaluation, grid search και paths.

- `runtime.py`
  Seed, επιλογή συσκευής, loading checkpoints και helpers για output folders.

- `data.py`
  Δημιουργία collocation points, initial-condition points και boundary-condition points.

- `problem.py`
  Η ακριβής λύση του benchmark, το PDE residual και οι βασικές μετρικές σφάλματος.

- `model.py`
  Ο ορισμός του μοντέλου `HeatPINN`.

- `training.py`
  Εκπαίδευση με Adam και fine-tuning με L-BFGS.

- `evaluation.py`
  Snapshot evaluation, global relative L2 evaluation και summary helpers.

- `plots.py`
  Όλα τα plots του project: histories, snapshots, slices, spacetime slices και sweep curves.

- `reporting.py`
  Excel writers για reports.

- `experiment_log.py`
  Γράφει experiment rows στο `runs/experiment_registry.xlsx`.


### Baseline parameters

- Domain: `t in [0, 1]`, `x in [-1, 1]`, `y in [-1, 1]`
- Sampling: `n_f=10000`, `n_i=1000`, `n_b=500`
- Model: `layer_sizes=(3, 20, 20, 20, 20, 20, 20, 20, 20, 1)`
- Adam: `epochs_adam=500`, `adam_lr=1e-3`
- Loss weights: `lambda_f=1.0`, `lambda_u=100.0`
- L-BFGS: `lbfgs_lr=1.0`, `lbfgs_max_iter=250`, `lbfgs_max_eval=312`

### Evaluation

- Snapshot times: `0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0`
- Slice times: `0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0`
- Fixed slice: `y=0.5`
- Snapshot grid: `100 x 100`
- Slice points: `200`
- Global evaluation: `80` spatial points και `11` time points

Η αξιολόγηση γίνεται πάντα σε σύγκριση με την ακριβή αναλυτική λύση του benchmark.

- Στα `snapshots`, για κάθε χρονική στιγμή φτιάχνω πλέγμα `100 x 100` στο `(x, y)` και μετράω:
  `relative_l2`, `mae`, `rmse`, `max_error`
- Στο `global evaluation`, παίρνω πυκνότερο πλέγμα στον χώρο και `11` χρονικά επίπεδα στο `[0, 1]`, και κρατάω:
  `global_mean_relative_l2`, `global_worst_relative_l2`
- Στο `train_heat_pinn.py` η ίδια διαδικασία τρέχει μία φορά μετά το Adam checkpoint και μία μετά το τελικό L-BFGS checkpoint
- Στο `evaluate_heat_pinn.py` γίνεται μόνο evaluation του τελικού συμβατού baseline checkpoint

### Grid search

Το default grid search είναι:

- `adam_epochs = (500, 2000, 5000)`
- `lbfgs_max_iter = (250, 500)`
- `lambda_f = (0.1, 0.3, 1.0, 3.0, 10.0)`
- `lambda_u = (10.0, 30.0, 100.0, 300.0, 1000.0)`
- `lbfgs_max_eval ~= 1.25 * lbfgs_max_iter`

Άρα συνολικά βγαίνουν `150` combinations.

## Εγκατάσταση

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Το project θέλει:

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `openpyxl`
- `tqdm`
- `torch`
- προαιρετικά `torch-directml` σε Windows

Αν δεν υπάρχει `torch-directml`, ο κώδικας θα τρέξει σε CPU. 

## Scripts

### 1. Baseline training

```powershell
python train_heat_pinn.py
```

Το script αυτό:

- φτιάχνει ένα νέο baseline run με timestamp
- δημιουργεί τα training points
- εκπαιδεύει το baseline μοντέλο με Adam
- αξιολογεί το Adam checkpoint και κρατάει τα αντίστοιχα plots/metrics
- κάνει fine-tuning με L-BFGS
- αξιολογεί το τελικό checkpoint
- γράφει baseline rows στο `runs/experiment_registry.xlsx`

Βασικά outputs:

- `runs/<baseline_run_name>/models/baseline_best_adam.pth`
- `runs/<baseline_run_name>/models/baseline_final_lbfgs.pth`
- `runs/<baseline_run_name>/reports/baseline_snapshot_metrics.xlsx`
- `runs/<baseline_run_name>/reports/baseline_global_summary.xlsx`
- `runs/<baseline_run_name>/figures/training_points.png`
- `runs/<baseline_run_name>/figures/baseline_best_adam_history.png`
- `runs/<baseline_run_name>/figures/baseline_best_adam_snapshots.png`
- `runs/<baseline_run_name>/figures/baseline_best_adam_slices.png`
- `runs/<baseline_run_name>/figures/baseline_best_adam_spacetime_slice.png`
- `runs/<baseline_run_name>/figures/baseline_final_lbfgs_history.png`
- `runs/<baseline_run_name>/figures/baseline_final_lbfgs_snapshots.png`
- `runs/<baseline_run_name>/figures/baseline_final_lbfgs_slices.png`
- `runs/<baseline_run_name>/figures/baseline_final_lbfgs_spacetime_slice.png`

### 2. Evaluation μόνο

```powershell
python evaluate_heat_pinn.py
```

Αυτό το script δεν κάνει training. Παίρνει το πιο πρόσφατο `baseline_final_lbfgs.pth` που ταιριάζει στο τωρινό baseline config και ξαναβγάζει evaluation reports και figures.

Σημαντικό:

- ψάχνει το πιο πρόσφατο συμβατό baseline checkpoint
- δεν φορτώνει αυτόματα grid-search winner
- δεν φορτώνει pruning checkpoint

Το evaluation run γράφεται σε νέο φάκελο της μορφής:

- `runs/<baseline_run_name>_evaluation_<timestamp>/...`

Βασικά outputs:

- `runs/<baseline_run_name>_evaluation_<timestamp>/reports/baseline_snapshot_metrics.xlsx`
- `runs/<baseline_run_name>_evaluation_<timestamp>/reports/baseline_global_summary.xlsx`
- `runs/<baseline_run_name>_evaluation_<timestamp>/figures/baseline_final_lbfgs_snapshots.png`
- `runs/<baseline_run_name>_evaluation_<timestamp>/figures/baseline_final_lbfgs_slices.png`
- `runs/<baseline_run_name>_evaluation_<timestamp>/figures/baseline_final_lbfgs_spacetime_slice.png`


### 3. Hyperparameter grid search

```powershell
python run_hyperparameter_grid.py
```

Χρήσιμα options:

```powershell
python run_hyperparameter_grid.py --cpu (εγώ το τρέχω έτσι γιατί υπάρχει κάποιο bug με την GPU memοry που δεν έχω λύσει ακόμη. Μπορεί να είναι και επειδή έχω AMD GPU)

python run_hyperparameter_grid.py --start-index 1 --end-index 10
```

Το script:

- φτιάχνει νέο grid-search run με timestamp
- τρέχει κάθε combination σαν ξεχωριστό baseline-style run
- κρατάει metrics μετά το Adam και μετά το τελικό L-BFGS
- γράφει progress reports όσο τρέχει
- στο τέλος γράφει και το τελικό ranking

Το καλύτερο model επιλέγεται με βάση το πόσο μικρό είναι το σφάλμα της πρόβλεψής (u_pred) του σε σχέση με την ακριβή λύση (u_exact) του benchmark

Πιο συγκεκριμένα, το ranking των combinations γίνεται:

- πρώτα με βάση το μικρότερο `global_mean_relative_l2`
- και σε ισοβαθμία με βάση το μικρότερο `global_worst_relative_l2`

Βασικά outputs:

- `results/grid_searches/<grid_run_name>/results.xlsx`
- `results/grid_searches/<grid_run_name>/summary.xlsx`
- `results/grid_searches/<grid_run_name>/best.xlsx`
- `runs/<combo_run_name>/...` για κάθε combination


## Τι περιέχουν τα βασικά Excel αρχεία

- `baseline_snapshot_metrics.xlsx`
  Αναλυτικά snapshot metrics ανά χρόνο.
  Περιέχει για κάθε `t` τις μετρικές:
  `relative_l2`, `mae`, `rmse`, `max_error`

- `baseline_global_summary.xlsx`
  Συνοπτικά global metrics του μοντέλου.
  Περιέχει τόσο τα μέσα / worst snapshot metrics όσο και τα:
  `global_mean_relative_l2`, `global_worst_relative_l2`

- `results/grid_searches/.../results.xlsx`
  Αναλυτικό grid-search report για όλα τα combinations, μαζί με Adam-stage και final-stage στοιχεία.
  Περιέχει hyperparameters, train times, Adam-stage metrics, final metrics και status για κάθε combination.

- `results/grid_searches/.../summary.xlsx`
  Το ranking των ολοκληρωμένων combinations με βάση τα τελικά global metrics.
  Το sorting γίνεται πρώτα με βάση το `global_mean_relative_l2` και μετά με βάση το `global_worst_relative_l2`.

- `results/grid_searches/.../best.xlsx`
  Η καλύτερη γραμμή του grid search.

- `runs/experiment_registry.xlsx`
  Συγκεντρωτικό αρχείο με experiment rows από `train_heat_pinn.py` και `run_hyperparameter_grid.py`.

## Πώς ονομάζονται τα runs

- Baseline run:
  `baseline_<adam_epochs>_<lbfgs_max_iter>-<lbfgs_max_eval>_<lambda_f>.<lambda_u>_<timestamp>`

- Evaluation run:
  `<baseline_run_name>_evaluation_<timestamp>`

- Grid-search run:
  `baseline_grid_search_<timestamp>`

## Notebook και scripts

Το `heat_eq.ipynb` είναι το αρχικό notebook όπου δοκίμαζα κομμάτια του κώδικα.

Αν τρέξει κάποιος όλα τα cells του notebook, κάνει baseline training και στη συνέχεια evaluation του τελικού baseline μοντέλου.




## Απολέσματα

Έχω τρέξει ήδη ένα πλήρες grid search με τα default parameters (`150` combinations).

Το τελικό συγκεντρωτικό αποτέλεσμα βρίσκεται στον φάκελο:

- `results/grid_searches/baseline_grid_search_20260330_022717/`

Τα βασικά αρχεία αυτού του run είναι:

- `results/grid_searches/baseline_grid_search_20260330_022717/results.xlsx`
- `results/grid_searches/baseline_grid_search_20260330_022717/summary.xlsx`
- `results/grid_searches/baseline_grid_search_20260330_022717/best.xlsx`

Το καλύτερο combination σε αυτό το run ήταν:

- `baseline_5000_500-625_1.100_20260330_022717_138`
- `adam_epochs=5000`
- `lbfgs_max_iter=500`
- `lbfgs_max_eval=625`
- `lambda_f=1.0`
- `lambda_u=100.0`
- `global_mean_relative_l2 ≈ 0.02817`
- `global_worst_relative_l2 ≈ 0.12445`

Το αντίστοιχο αναλυτικό run folder είναι:

- `example_runs/` για μερικά αντιπροσωπευτικά δείγματα runs (λόγω μεγέθους του κανονικού φακέλου)
- `results/grid_searches/baseline_grid_search_20260330_022717/` για τα συγκεντρωτικά αποτελέσματα όλου του grid search

