# AMES GNN-MTL: Graph Neural Network Multi-Task Learning for Ames Mutagenicity Prediction

This codebase trains and evaluates a GNN-based multi-task learning model to predict Ames mutagenicity across five bacterial strains (TA98, TA100, TA102, TA1535, TA1537) using the ISSSTY dataset. Molecules are represented as graphs (GINEConv architecture); a shared graph encoder feeds into five task-specific prediction heads.

---

## Directory Structure

```
AMES_FINAL/
├── run_model.py                     # Main driver: train, evaluate, HP optimization, analysis
├── GNN_explainer_analysis.py        # GNNExplainer + Integrated Gradients analysis
├── train_sample.yml                 # Example training configuration
├── data.csv                         # Molecular data with SMILES and labels
├── plan.md                          # Architecture and implementation notes
│
├── checkpoints/                     # Saved model checkpoints (.pt files)
├── optuna/                          # Optuna study files (.pkl)
│
├── BuildNN_GNN_MTL_GINEConv.py      # Model architecture (GINEConv GNN + MTL heads)
├── callbacks.py                     # EarlyStopping, LRScheduler, UserStopping
├── compute_metrics.py               # Metrics: sensitivity, specificity, MCC, balanced accuracy
├── data.py                          # Data loading utilities
├── load_data.py                     # Molecular descriptor loading
├── graph_dataset.py                 # PyTorch Geometric dataset wrapper
├── MTLDataset.py                    # Multi-task dataset class
├── masked_loss_function.py          # BCE loss with masking for missing labels (-1)
├── set_seed.py                      # Random seed utilities
├── TaskSpecificGNN.py               # Task-specific head module
├── features.py                      # Atom/bond feature definitions
├── generate_graphs.py               # Graph generation from XYZ files
├── graph_maker.py                   # Graph construction utilities
└── [other helper modules]
```

The graph database lives outside this directory, and is created using generate_graphs.py:

```
GraphDataBase_AMES/
├── train/           # .pkl graph files for training molecules
├── validate/        # .pkl graph files for validation molecules
├── test/            # .pkl graph files for test molecules
└── graph_description.yml
```

---

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n ames_gnn python=3.13
conda activate ames_gnn
```

### 2. Install PyTorch

```bash
pip install torch==2.8.0
```

### 3. Install PyTorch Geometric

```bash
pip install torch_geometric==2.6.1
```

### 4. Install remaining dependencies

```bash
pip install \
    rdkit==2025.3.5 \
    optuna==4.2.0 \
    seaborn==0.13.2 \
    matplotlib==3.10.8 \
    numpy==2.4.2 \
    pandas==3.0.0 \
    scikit-learn==1.8.0 \
    joblib==1.5.3 \
    tensorboard==2.20.0 \
    PyYAML==6.0.2 \
    networkx==3.6.1 \
    h5py==3.14.0 \
    Pillow==12.1.1 \
    markdown==3.8 \
    iterative-stratification
```

### 5. Prepare the graph database

The model reads pre-computed molecular graphs from `GraphDataBase_AMES/`. If graphs do not yet exist for your molecules, generate them:

```bash
python graph_maker.py graph_maker_sample.yml
```

The database path is set in the YAML config file (`database` key). The default points to:

```
/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/GraphDataBase_AMES/
```

Update this path in your YAML if your database is in a different location.

---

## Configuration File (YAML)

All training modes read hyperparameters from a YAML file (see `train_sample.yml` for a complete example). Key fields:

| Field | Description |
|-------|-------------|
| `database` | Path to the graph database directory |
| `data_file` | Path to `data.csv` with SMILES, labels, and split assignments |
| `nGraphConvolutionLayers` | Number of GINEConv layers |
| `nNodeNeurons` / `nEdgeNeurons` | Hidden dimensions for node/edge features |
| `nSharedLayers` / `nTargetSpecificLayers` | Depth of shared and task-specific heads |
| `nShared` / `nTarget` | Neuron counts per layer (list) |
| `dropoutGNN` / `dropoutShared` / `dropoutTarget` | Dropout rates |
| `w1`–`w5` | Per-task loss weights |
| `ActivationFunction` | e.g. `"Tanh"` |
| `nEpochs` | Maximum training epochs |
| `nBatch` | Batch size |
| `learningRate` | Initial learning rate |
| `L2Regularization` | Weight decay coefficient |
| `weightedCostFunction` | Whether to use weighted BCE loss |
| `callbacks` | `earlyStopping`, `LRScheduler`, `UserStopping` sub-sections |

---

## Running the Model (`run_model.py`)

All modes share the pattern:

```bash
python run_model.py --mode <mode> --output_dir <dir> [--input_file <yaml>] [options]
```

### `train` — Train with fixed hyperparameters

Trains the model using HP from the YAML file. Saves checkpoints and TensorBoard logs.

```bash
python run_model.py \
    --mode train \
    --input_file train_sample.yml \
    --output_dir ./output/train_run1 \
    --checkpoints_dir ./checkpoints
```

**Outputs:**
- `checkpoints/checkpoint_epoch_N.pt` — model checkpoint at each epoch
- `output/train_run1/tensorboard/` — TensorBoard event files

Per-epoch output is printed to the console:
```
Epoch 1/100  train_loss=0.523411  val_loss=0.498762  lr=2.70e-04
```
Early stopping and LR reduction events are also logged when they occur.

### `hp_opt` — Hyperparameter optimization with Optuna

Runs Optuna TPE search using 5-fold CV on the training set. Saves a study `.pkl` file that can be resumed.

```bash
python run_model.py \
    --mode hp_opt \
    --input_file train_sample.yml \
    --output_dir ./output/hp_search \
    --optuna_dir ./optuna \
    --n_trials 100
```

To resume an existing study:

```bash
python run_model.py \
    --mode hp_opt \
    --input_file train_sample.yml \
    --output_dir ./output/hp_search \
    --optuna_file ./optuna/study_20260101.pkl \
    --n_trials 50
```

**Outputs:**
- `optuna/study_YYYYMMDD.pkl` — Optuna study (auto-named by date if `--optuna_file` not given)

### `seeds_cfv` — 5-fold cross-validation across multiple seeds

Runs 5-fold CV for each random seed. Useful for assessing model stability and selecting the best seed/fold for final evaluation.

```bash
python run_model.py \
    --mode seeds_cfv \
    --input_file train_sample.yml \
    --output_dir ./output/cfv_results \
    --checkpoints_dir ./checkpoints \
    --seeds "3 7 15 24 42 45 62 77 79 88 90"
```

**Outputs (in `--output_dir`):**
- `metrics_{seed}_{fold}.csv` — per-fold classification metrics
- `avg_val_losses.csv` — average validation loss per seed across 5 folds
- `val_losses.csv` — validation loss for every seed × fold combination

Checkpoints are saved to `--checkpoints_dir/metrics_{seed}_{fold}.pt`.

### `eval` — Evaluate a single checkpoint on the test set

Loads a checkpoint and evaluates on the test set. ROC curves and Precision-Recall curves are always saved. By default uses a threshold of 0.5; use `--use_thresholds` to optimise thresholds on the validation set.

```bash
# Default: 0.5 threshold for all tasks
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/metrics_45_1.pt

# Temperature scaling + per-task threshold optimisation (maximise sensitivity)
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/metrics_45_1.pt \
    --use_thresholds --temperature_scaling --threshold_metric sn

# Temperature scaling + single consensus threshold (maximise balanced accuracy)
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/metrics_45_1.pt \
    --use_thresholds --temperature_scaling --tune_consensus_threshold --threshold_metric bal_acc
```

**Threshold flags (only active when `--use_thresholds` is set):**

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold_metric` | `sn` | Metric to maximise: `sn`, `sp`, `bal_acc`, `ppv`, `npv`, `mcc`, `f1`, `h` |
| `--tune_consensus_threshold` | off | Optimise one shared threshold for the consensus (OR) outcome instead of 5 per-task thresholds |
| `--temperature_scaling` | off | Fit scalar temperature T on val set (minimises NLL) before thresholding. Can also be used without `--use_thresholds` to calibrate probabilities. |

**Outputs:**
- `metrics.csv` — per-strain metrics (TP, TN, FP, FN, Sp, Sn, PPV, NPV, Acc, Bal acc, MCC, F1 score, H score)
- `metrics_cons.csv` — consensus (OR rule) metrics
- `misclassified_files.csv` — molecules where consensus prediction was wrong
- `model_output_raw.csv` — probabilities, true labels, binary predictions, and consensus per molecule
- `roc_curves.png` — ROC curves for each strain + consensus (AUC annotated)
- `pr_curves.png` — Precision-Recall curves for each strain + consensus (AP annotated)

### `top_seeds_eval` — Evaluate top N seeds and average metrics

Reads `val_losses.csv` (from `seeds_cfv`), selects the top `N` seeds by lowest average validation loss, loads their per-fold checkpoints, evaluates each on the test set, and averages results. Supports the same `--use_thresholds`, `--temperature_scaling`, and `--tune_consensus_threshold` flags as `eval`.

```bash
# Default: 0.5 threshold
python run_model.py \
    --mode top_seeds_eval \
    --input_file train_sample.yml \
    --output_dir ./output/top_seeds \
    --metrics_dir ./output/cfv_results \
    --checkpoints_dir ./checkpoints \
    --n_top_seeds 5

# With temperature scaling + threshold optimisation
python run_model.py \
    --mode top_seeds_eval \
    --input_file train_sample.yml \
    --output_dir ./output/top_seeds \
    --metrics_dir ./output/cfv_results \
    --checkpoints_dir ./checkpoints \
    --n_top_seeds 5 \
    --use_thresholds --temperature_scaling --threshold_metric sn
```

**Outputs:**
- `top_seeds_all_metrics.csv` — per-seed, per-fold test metrics
- `top_seeds_avg_metrics.csv` — metrics averaged across top seeds and folds

### `analyze_cfv` — Plot and summarize cross-validation results

Reads per-fold metric CSVs from a completed `seeds_cfv` run and generates summary plots. Identifies the top 5 seeds by lowest average validation loss.

```bash
python run_model.py \
    --mode analyze_cfv \
    --output_dir ./output/cfv_plots \
    --metrics_dir ./output/cfv_results
```

**Outputs (PNG files in `--output_dir`):**
- `metrics_barplot.png` — per-strain metrics bar chart
- `mcc_barplot.png` — MCC per strain
- `validation_loss_heatmap.png` — heatmap of validation loss for every seed × fold
- `top_seeds_by_val_loss.csv` — top 5 seeds ranked by average validation loss

### `viz_optuna` — Visualize an Optuna study

Generates optimization history and hyperparameter importance plots for one or all studies in a directory.

Single study:

```bash
python run_model.py \
    --mode viz_optuna \
    --output_dir ./output/optuna_plots \
    --optuna_file ./optuna/study_20260101.pkl
```

All studies in a directory:

```bash
python run_model.py \
    --mode viz_optuna \
    --output_dir ./output/optuna_plots \
    --optuna_dir ./optuna
```

**Outputs:**
- `optimization_history.png`
- `param_importances.png`
- `val_loss_history.png`

---

## GNNExplainer Analysis (`GNN_explainer_analysis.py`)

Runs GNNExplainer to identify important molecular fragments, and optionally runs Integrated Gradients for input feature importance.

```bash
python GNN_explainer_analysis.py \
    --input_file train_sample.yml \
    --output_dir ./output/explainer \
    --checkpoint_file ./checkpoints/metrics_45_1.pt
```

To also run Integrated Gradients input feature importance:

```bash
python GNN_explainer_analysis.py \
    --input_file train_sample.yml \
    --output_dir ./output/explainer \
    --checkpoint_file ./checkpoints/metrics_45_1.pt \
    --analyze_input_features
```

To override the data file path from the YAML:

```bash
python GNN_explainer_analysis.py \
    --input_file train_sample.yml \
    --output_dir ./output/explainer \
    --checkpoint_file ./checkpoints/metrics_45_1.pt \
    --data_file /path/to/custom_data.csv
```

**Outputs:**
- PDF reports with fragment importance visualizations per task
- CSV files with atom/fragment importance scores
- (with `--analyze_input_features`) Bar charts and heatmaps of node feature importances

---

## Visualizing Training with TensorBoard

TensorBoard logs are written to `<output_dir>/tensorboard/` during `train` mode. To view them:

```bash
tensorboard --logdir ./output/train_run1/tensorboard
```

Then open `http://localhost:6006` in a browser. The `Loss/train` and `Loss/val` scalars are logged at every epoch.

---

## Stopping Training Early (UserStopping)

If `UserStopping` is listed in the `callbacks` section of the YAML, a `STOPFLAG.yml` file is created at the start of training. To stop training gracefully at the end of the current epoch, edit this file and set:

```yaml
STOPFLAG: True
```

The model will save a checkpoint and exit cleanly.

---

## Data File Format (`data.csv`)

The data file should contain at minimum:

| Column | Description |
|--------|-------------|
| `SMILES` | Canonical SMILES string |
| `TA98`, `TA100`, `TA102`, `TA1535`, `TA1537` | Binary labels (0 = negative, 1 = positive, -1 = missing) |
| `Overall` | Overall consensus label (used in `eval` mode for misclassification analysis) |
| `split` | `train`, `validate`, or `test` |

---

## Key References

- **Architecture**: GINEConv-based GNN encoder with shared + task-specific MLP heads
- **Loss**: Masked weighted binary cross-entropy (ignores `-1` labels)
- **CV**: MultilabelStratifiedKFold (5 folds) preserving label distribution
- **Threshold optimization**: Coordinate ascent with 1-SE rule on validation set
- **Consensus prediction**: OR rule across all 5 task heads
