# A Multitask Graph Neural Network Framework for Ames Mutagenicity Prediction
Abigail E. Teitgen, Eugenia Ulzurrun, Nuria E. Campillo, and Eduardo R. Hernández

This codebase trains and evaluates a GNN-based multi-task learning model to predict Ames mutagenicity across five bacterial strains (TA98, TA100, TA102, TA1535, TA1537). Molecules are represented as graphs (GINEConv architecture).

---

## Quickstart

Follow these steps to reproduce the main results. See [Setup](#setup) first to install dependencies.

**Build the graph database**

Unzip the provided XYZ files, then build the graph database:

```bash
unzip XYZ_files.zip -d ./FILES_XYZ
python graph_maker.py graph_maker_sample.yml
```

Update `DataBaseDirectory`, `TargetDirectory`, and `DataPath` in `graph_maker_sample.yml` to point to your local paths before running.

**Train the model**

```bash
python run_model.py \
    --mode train \
    --input_file train_sample.yml \
    --output_dir ./output/train_run1 \
    --checkpoints_dir ./checkpoints/train_run1 \
    --use_thresholds --temperature_scaling --threshold_metric bal_acc
```

Update `database` and `data_file` in `train_sample.yml` to match your graph database and data paths.

**Evaluate the model**

```bash
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt \
    --use_thresholds --temperature_scaling --threshold_metric bal_acc
```

Update `database` and `data_file` in `train_sample.yml` to match your graph database and data paths.
Replace `metrics_77_0.pt` with whichever checkpoint you want to analyze (metrics_77_0.pt is the checkpoint used in the paper, i.e. best model weights after crossfold validation, but you can also use a checkpoint from a training run etc.).


**Plot and summarize cross-validation results**

```bash
python run_model.py \
    --mode analyze_cfv \
    --output_dir ./output/cfv_plots \
    --metrics_dir ./metrics/final/
```

**Run the explainer analysis**

```bash
python GNN_explainer_analysis.py \
    --input_file train_sample.yml \
    --output_dir ./output/explainer \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt \
    --analyze_input_features
```

Replace `metrics_77_0.pt` with whichever provided checkpoint you want to analyze (metrics_77_0.pt is the checkpoint used in the paper).

Note: Only metrics_77_0.pt is provided in the checkpoints/final directory, due to size constraints on GitHub. This checkpoint represents the best model weights from crossfold validation analysis and was utilized for all explainer analysis etc. To reproduce all checkpoints, rerun crossfold validation analysis (see section below)

---




## Code overview

```
AMES_FINAL/
├── run_model.py                     # Main driver: train, evaluate, HP optimization, analysis
├── GNN_explainer_analysis.py        # GNNExplainer + Integrated Gradients + novel-fragment analysis
├── shap_analysis_standalone.py      # Standalone grouped-KernelSHAP feature analysis (this requires a separate venv)
├── graph_maker.py                   # Build graph database from XYZ files
├── train_sample.yml                 # Example training/evaluation configuration
├── graph_maker_sample.yml           # Example graph construction configuration
│
├── BuildNN_GNN_MTL_GINEConv.py      # Model architecture (GINEConv GNN + MTL heads)
├── TaskSpecificGNN.py               # Task-specific head wrapper (used by GNN explainer)
├── callbacks.py                     # EarlyStopping, LRScheduler, UserStopping
├── compute_metrics.py               # Metrics: sensitivity, specificity, MCC, balanced accuracy
├── masked_loss_function.py          # BCE loss with masking for missing labels (-1)
├── graph_dataset.py                 # PyTorch Geometric dataset wrapper
├── MTLDataset.py                    # Multi-task dataset class
├── load_data.py                     # Molecular data loading and split assignment
├── data.py                          # Data loading utilities
├── set_seed.py                      # Random seed utilities
├── device.py                        # Device selection (CPU/CUDA)
├── features.py                      # RBF distance feature definitions
├── generate_graphs.py               # Graph generation loop from XYZ files
├── atomic_structure_graphs.py       # Abstract graph base class
├── set_up_atomic_structure_graphs.py# Factory for graph construction strategy
├── XG_graphs.py                     # XG graph implementation
├── count_model_parameters.py        # Parameter count utility
├── exceptions.py                    # Custom exceptions
│
├── smiles_to_xyz.py                 # Convert SMILES → 3D XYZ files
├── calculate_descriptors.py         # Compute Mordred 2D descriptors
├── build_graphs_new_dataset.py      # Graph building for additional datasets
├── counter.py                       # Count atom species present in XYZ database
├── count_species.py                 # Species counting helper
├── visualize_graphs.py              # Render molecular graphs side-by-side with 2D structures
├── ISSSTY_utils.py                  # ISSSTY dataset utilities
├── structure_utils.py               # Molecular structure helpers
├── requirements_shap.txt            # Requirements for venv for SHAP analysis
│
└── STOPFLAG.yml                     # Set STOPFLAG: True to halt training gracefully
```

The graph database is located outside this directory, and is created using `graph_maker.py`:

```
GraphDataBase_AMES/
├── train/                 # .pkl graph files for training molecules
├── validate/              # .pkl graph files for validation molecules
├── test/                  # .pkl graph files for test molecules
└── graph_description.yml  # Auto-generated; records feature counts and construction params
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
    scipy \
    joblib==1.5.3 \
    tensorboard==2.20.0 \
    PyYAML==6.0.2 \
    networkx==3.6.1 \
    h5py==3.14.0 \
    Pillow==12.1.1 \
    markdown==3.8 \
    mendeleev \
    mordredcommunity \
    iterative-stratification \
    openpyxl \
    "setuptools<70.0.0"
```

> **Notes:**
> - `mendeleev` is required by the graph construction modules to look up element properties (period, block, electronegativity, etc.) used as node features.
> - `mordred` is only needed when running `calculate_descriptors.py` or using `inputMode: "descriptor"`/`"combined"`.
> - the SHAP analysis requires a separate venv because of package version conflicts, and is set up as a standalone analysis, see section below 

---


## Data File Format (`Ames_mutagenicity_strain_specific.csv`)

The data file should contain at minimum:

| Column | Description |
|--------|-------------|
| `SMILES` | Canonical SMILES string |
| `TA98`, `TA100`, `TA102`, `TA1535`, `TA1537` | Binary labels (0 = negative, 1 = positive, -1 = missing) |
| `Overall` | Overall consensus label |
| `split` | `train`, `validate (internal)`, or `test (external)` |

For `"descriptor"` and `"combined"` input modes, the file must also contain Mordred descriptor columns — generate these with `calculate_descriptors.py`. This is only used for ablation analysis.

---

## Workflow Overview

```
SMILES CSV
    │
    ├─ smiles_to_xyz.py          →  XYZ files (3D conformers)
    │
    ├─ counter.py                →  atom species summary (to configure graph_maker_sample.yml)
    │
    ├─ graph_maker.py            →  graph database (.pkl files + graph_description.yml)
    │
    ├─ calculate_descriptors.py  →  CSV with Mordred descriptor columns (optional)
    │
    ├─ run_model.py hp_opt       →  hyperparameter search (Optuna)
    ├─ run_model.py train        →  train with fixed hyperparameters
    ├─ run_model.py seeds_cfv    →  cross-validation across seeds
    ├─ run_model.py eval         →  evaluate checkpoint on test set
    │
    └─ GNN_explainer_analysis.py →  structural alert overlap + feature importance
```

---


## Step 1: Convert SMILES to XYZ (`smiles_to_xyz.py`)

**Note: The XYZ files for the dataset used in this paper are provided in XYZ_files.zip, so you do not need to re-run this step**

Generates 3D XYZ files from a SMILES CSV using RDKit ETKDGv3 conformer generation with MMFF/UFF optimization.

**Multi-component SMILES (salts/counter-ions):** The script automatically keeps only the largest fragment (by heavy-atom count) when a SMILES contains multiple disconnected components. Single-atom fragments are retained.

```bash
python smiles_to_xyz.py \
    --input_csv data.csv \
    --smiles_col SMILES \
    --output_dir ./FILES_XYZ
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_csv` | `data.csv` | Input CSV with SMILES |
| `--smiles_col` | `SMILES` | Name of the SMILES column |
| `--output_dir` | `./FILES_XYZ` | Directory to write XYZ files |

**Output:** One `{row}_ames_mutagenicity_data_{row}.xyz` file per molecule.

---

## Step 2: Count Atom Species (`counter.py`)

**Note: This has already been done for the current dataset and correctly updated in graph_maker_sample.yml, so you do not need to rerun this step**

Before building graphs, run this to see which elements are present in the XYZ database. Use the output to set `species` and `nMaxNeighbours` in `graph_maker_sample.yml`.

```bash
python counter.py graph_maker_sample.yml
```

Reads `DataBaseDirectory` from the YAML file and prints an element-count summary to stdout.

---

## Step 3: Build the Graph Database (`graph_maker.py`)

Converts XYZ files to PyTorch Geometric graph objects and writes a `graph_description.yml` into the target directory that `run_model.py` uses to determine feature dimensions automatically.

```bash
python graph_maker.py graph_maker_sample.yml
```


### Graph Construction Configuration (`graph_maker_sample.yml`)
**Note: The current graph_maker_sample.yml file will generate the graphs used in the paper, so you don't need to change any values unless desired to reproduce ablation analysis etc. The only things necessary to update are the paths.**

| Field | Description |
|-------|-------------|
| `DataBaseDirectory` | Path to source XYZ files |
| `TargetDirectory` | Path to store output `.pkl` graph files |
| `DataPath` | Path to the CSV file used to assign train/val/test splits |
| `graphType` | Graph construction style (default `"XG"` — covalent-radius based) |
| `nodeFeatures` | List of Mendeleev property keywords for node features |
| `species` | List of chemical elements present in the dataset |
| `nMaxNeighbours` | Maximum number of neighbours per atom (set from `counter.py` output) |
| `BondAngleFeatures` | `True`/`False` — include bond angle cosine sums as edge features |
| `DihedralAngleFeatures` | `True`/`False` — include dihedral angle cosine sums as edge features |
| `distanceEncoding` | `"raw"` (single float, default) or `"rbf"` (Gaussian RBF expansion) |
| `RBFParameters` | RBF encoding parameters: `n_features`, `r_min`, `r_max`, `sigma` |
| `generate_graphs` | Set `False` to skip graph generation (useful for debugging the YAML) |

### Distance Encoding

By default, each edge stores the raw bond distance as a single float. With `distanceEncoding: "rbf"`, distances are expanded into a vector of Gaussian radial basis functions:

```
u_k(d) = exp(-(d - mu_k)^2 / sigma^2)
```

where the centers `mu_k` are evenly spaced between `r_min` and `r_max`.

Example RBF configuration:

```yaml
distanceEncoding: "rbf"
RBFParameters:
  n_features: 20
  r_min: 0.0
  r_max: 5.0
  sigma: 0.5
```

> **Important:** Graphs must be regenerated when changing `distanceEncoding` or toggling `BondAngleFeatures`/`DihedralAngleFeatures`, as the edge feature dimension changes. The model reads the correct dimension from `graph_description.yml` automatically.

---

## Step 4 (Optional): Compute Mordred Descriptors (`calculate_descriptors.py`)

Only needed for `inputMode: "descriptor"` or `"combined"`. Computes all 2D Mordred descriptors and appends them as columns to the CSV.

```bash
python calculate_descriptors.py \
    --input_csv Ames_mutagenicity_strain_specific.csv \
    --output_csv Ames_mutagenicity_strain_specific_descriptors.csv
```

The resulting file is used with `run_model.py` by setting `data_file` in the training YAML.
This is only necessary to reproduce comparison analysis to Martinez et al. 

---

## Visualizing the Graph Database (`visualize_graphs.py`)

Renders each molecular graph side-by-side with the 2D chemical structure to verify atom types, connectivity, and bond distances.

```bash
python visualize_graphs.py \
    --input_file train_sample.yml \
    --n_graphs 100 \
    --partition test \
    --output_dir ./graph_visualization \
    --output_format pdf
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | `train_sample.yml` | YAML config (reads `database` and `data_file` paths) |
| `--database_dir` | from YAML `database` | Path to graph database root |
| `--data_file` | from YAML `data_file` | CSV with SMILES and labels |
| `--n_graphs` | `20` | Number of graphs to visualize |
| `--partition` | `test` | `train`, `validate`, `test`, or `all` |
| `--output_dir` | `./graph_viz` | Directory to save output |
| `--output_format` | `pdf` | `pdf` (all in one file) or `png` (one file per graph) |
| `--show_H` | off | Include hydrogen atoms in the graph panel |

Each figure shows two panels: RDKit 2D structure on the left and the molecular graph on the right, with nodes colored by element, edge color indicating bond distance, and labels showing mol ID and per-strain toxicity labels.

**Outputs:**
- `pdf` mode: `graphs.pdf` — one page per molecule
- `png` mode: `fig_{mol_id}.png` per molecule

---



## Training Configuration (`train_sample.yml`)

All `run_model.py` modes read hyperparameters and paths from a YAML file. train_sample includes the default parameters from the paper. Key fields:

| Field | Description |
|-------|-------------|
| `database` | Path to the graph database directory |
| `data_file` | Path to CSV with SMILES, labels, split assignments (and optionally descriptors) |
| `nGraphConvolutionLayers` | Number of GINEConv layers |
| `nNodeNeurons` / `nEdgeNeurons` | Hidden dimensions for node/edge features |
| `nSharedLayers` / `nTargetSpecificLayers` | Depth of shared and task-specific heads |
| `nShared` / `nTarget` | Neuron counts per layer (list) |
| `dropoutGNN` / `dropoutShared` / `dropoutTarget` | Dropout rates |
| `momentumBatchNorm` | Batch normalization momentum |
| `w1`–`w5` | Per-task loss weights |
| `ActivationFunction` | e.g. `"Tanh"` |
| `nEpochs` | Maximum training epochs |
| `nBatch` | Batch size |
| `learningRate` | Initial learning rate |
| `L2Regularization` | Weight decay coefficient |
| `weightedCostFunction` | Whether to use weighted BCE loss |
| `inputMode` | `"gnn"` (default), `"descriptor"`, or `"combined"` |
| `callbacks` | `earlyStopping`, `LRScheduler`, `UserStopping` sub-sections |

---

## Input Modes

| Mode | Description | Data requirement |
|------|-------------|-----------------|
| `"gnn"` | Graph features only via GINEConv layers (default) | Graph database |
| `"descriptor"` | Mordred 2D molecular descriptors only | CSV with descriptor columns |
| `"combined"` | GNN graph embedding concatenated with descriptor vector | Both |

---

## Running the Model (`run_model.py`)

All modes share the pattern:

```bash
python run_model.py --mode <mode> --output_dir <dir> [--input_file <yaml>] [options]
```

### `train` — Train with fixed hyperparameters
**Note: The current train_sample.yml file is set up with the optimized hyperparameters from the paper, so you will just need to update the paths.**

```bash
python run_model.py \
    --mode train \
    --input_file train_sample.yml \
    --output_dir ./output/train_run1 \
    --checkpoints_dir ./checkpoints/train_run_1
```

**Outputs:**
- `checkpoints/final/checkpoint_epoch_N.pt` — model checkpoint at each epoch
- `output/train_run1/tensorboard/` — TensorBoard event files

---

## Visualizing Training with TensorBoard

TensorBoard logs are written to `<output_dir>/tensorboard/` during `train` mode.

```bash
tensorboard --logdir ./output/train_run1/tensorboard
```

Then open `http://localhost:6006`. The `Loss/train` and `Loss/val` scalars are logged at every epoch.

---

## Stopping Training Early (UserStopping)

If `UserStopping` is listed in the `callbacks` section of the YAML, a `STOPFLAG.yml` file is created at the start of training. To stop training at the end of the current epoch, set:

```yaml
STOPFLAG: True
```

The model will save a checkpoint and exit cleanly.

---



### `hp_opt` — Hyperparameter optimization with Optuna

Runs Optuna search using 5-fold crossfold validation on the training set. The study is saved after every completed trial so progress is preserved if interrupted.

```bash
python run_model.py \
    --mode hp_opt \
    --input_file train_sample.yml \
    --output_dir ./output/hp_search \
    --optuna_dir ./optuna \
    --n_trials 100
```

To seed the first trial with the YAML hyperparameters (gives the optimizer a strong baseline):

```bash
python run_model.py ... --seed_params
```

**Outputs:**
- `optuna/study_YYYYMMDD.pkl` — Optuna study (auto-named by date if `--optuna_file` not given)

---



### `seeds_cfv` — 5-fold cross-validation across multiple seeds

```bash
python run_model.py \
    --mode seeds_cfv \
    --input_file train_sample.yml \
    --output_dir ./metrics/final \
    --checkpoints_dir ./checkpoints/final \
    --seeds "3 7 15 24 42 45 62 77 79 88 90"
```

**Outputs (in `--output_dir`):**
- `metrics_{seed}_{fold}.csv` — per-fold classification metrics
- `avg_val_losses.csv` — average validation loss per seed across 5 folds
- `val_losses.csv` — validation loss for every seed × fold

Checkpoints saved to `--checkpoints_dir/metrics_{seed}_{fold}.pt`.

---



### `eval` — Evaluate a single checkpoint on the test set

```bash
# Default: 0.5 threshold for all tasks
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt

# Temperature scaling + per-task threshold optimisation (maximise sensitivity)
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt \
    --use_thresholds --temperature_scaling --threshold_metric sn

# Temperature scaling + single consensus threshold (maximise balanced accuracy)
python run_model.py \
    --mode eval \
    --input_file train_sample.yml \
    --output_dir ./output/eval_results \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt \
    --use_thresholds --temperature_scaling --tune_consensus_threshold --threshold_metric bal_acc
```

**Threshold flags (only active when `--use_thresholds` is set):**

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold_metric` | `sn` | Metric to maximise: `sn`, `sp`, `bal_acc`, `ppv`, `npv`, `mcc`, `f1`, `h` |
| `--tune_consensus_threshold` | off | Optimise one shared threshold for the consensus outcome |
| `--temperature_scaling` | off | Fit scalar temperature T on val set before thresholding; can also be used without `--use_thresholds` to calibrate probabilities |

**Outputs:**
- `metrics.csv` — per-strain metrics (TP, TN, FP, FN, Sp, Sn, PPV, NPV, Acc, Bal acc, MCC, F1, H)
- `metrics_cons.csv` — consensus metrics
- `misclassified_files.csv` — molecules where consensus prediction was wrong
- `model_output_raw.csv` — probabilities, true labels, binary predictions, and consensus per molecule
- `roc_curves.png` — ROC curves for each strain + consensus (AUC annotated)
- `pr_curves.png` — Precision-Recall curves for each strain + consensus (AP annotated)
- `roc_curve_consensus.svg` / `pr_curve_consensus.svg` — standalone single-panel consensus-only ROC / PR curves

---



### `top_seeds_eval` — Evaluate top N seeds and average metrics

Reads `val_losses.csv` from `seeds_cfv`, selects the top N seed/fold pairs, and averages test set metrics across them. The selection procedure is:

1. For each seed, compute the **average validation loss across all 5 folds**.
2. Rank seeds by that average; keep the top N (lowest loss).
3. Within each top seed, pick the **single best fold** (lowest individual fold loss).
4. Evaluate those N (seed, fold) pairs and average their test metrics.

Supports the same `--use_thresholds`, `--temperature_scaling`, and `--tune_consensus_threshold` flags as `eval`.

```bash
python run_model.py \
    --mode top_seeds_eval \
    --input_file train_sample.yml \
    --output_dir ./output/top_seeds \
    --metrics_dir ./metrics/final/ \
    --checkpoints_dir ./checkpoints/final/ \
    --n_top_seeds 5
```

**Outputs:**
- `top_seeds_all_metrics.csv` — per (seed, fold) test metrics
- `top_seeds_avg_metrics.csv` — metrics averaged across the top seed/fold pairs

### `analyze_cfv` — Plot and summarize cross-validation results

```bash
python run_model.py \
    --mode analyze_cfv \
    --output_dir ./output/cfv_plots \
    --metrics_dir ./metrics/final/
```

**Outputs:**
- `metrics_barplot.png` — per-strain metrics bar chart
- `mcc_barplot.png` — MCC per strain
- `validation_loss_heatmap.png` — heatmap of validation loss for every seed × fold
- `top_seeds_by_val_loss.csv` — top 5 seeds ranked by average validation loss




---

## GNNExplainer + Feature Importance Analysis (`GNN_explainer_analysis.py`)

Runs GNNExplainer to identify important molecular fragments, computes structural alert overlap scores, and mines recurring **novel** substructures (enriched in mutagenic predictions but not matching known alerts).

```bash
# GNNExplainer
python GNN_explainer_analysis.py \
    --input_file train_sample.yml \
    --output_dir ./output/explainer \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt
```

Replace `metrics_77_0.pt` with whichever provided checkpoint you want to analyze (`metrics_77_0.pt` is the checkpoint used in the paper).

**Outputs (in `--output_dir`):**

*Structural alert analysis (GNNExplainer path):*
- `alert_instance_grids/` — per-alert grid images showing each matching molecule with three-color atom highlighting:
  - **Orange** = atoms in both the SMARTS match and the GNN important set (overlap)
  - **Blue** = alert atoms not identified as important by the GNN
  - **Red** = GNN-important atoms not in the alert
- `alert_averaged_plots_positional/` — per-alert positional averaging heatmaps showing which SMARTS atom positions are most frequently GNN-important across all matching molecules. Up to 3 representative molecules per alert:
  - `<alert>_smarts_pos_avg_rep0.png` — smallest matching molecule
  - `<alert>_smarts_pos_avg_rep1.png` — second smallest
  - `<alert>_smarts_pos_avg_rep2.png` — third smallest
  - `attribution_color_scale.svg` — vertical colour-scale legend (relative attribution, white→dark red) for these positional plots
- `toxic_overlap_by_strain_heatmap.pdf` — heatmap of mean GNN overlap score (toxic molecules only) per alert per strain; alerts with zero overall overlap excluded
- `alert_auc_by_strain_heatmap.pdf` — heatmap of AUROC per alert per strain (model predicted probability vs ground-truth mutagenicity for alert-matched molecules; requires ≥10 samples per cell)
- `alert_performance_bars.pdf` — horizontal bar chart of mean overlap score per alert, sorted descending; alerts with zero overlap excluded
- `alert_category_auc_heatmap.pdf` — two-panel figure per alert: (left) fraction of molecules in each of four categories; (right) one-vs-rest AUROC per category. Categories are defined relative to each alert:
  - **A** = has alert + mutagenic
  - **B** = has alert + not mutagenic
  - **C** = no alert + mutagenic
  - **D** = no alert + not mutagenic
- `alert_category_auc_summary.csv` — table of fractions and AUROCs per category for each alert
- `overlap_diagnostic.xlsx` — per-molecule diagnostic with raw SMARTS atom counts, expanded atom counts, GNN overlap counts, and computed overlap scores
- `summary_rows/` — molecule summary PDFs, one row per molecule (5 strain cells + consensus + alert overview), important GNN atoms highlighted in **uniform red**:
  - `summary_correct_toxic.pdf` — correctly predicted mutagenic molecules
  - `summary_correct_nontoxic.pdf` — correctly predicted non-mutagenic molecules
  - `summary_incorrect.pdf` — incorrectly predicted molecules
- `summary_rows_scaled/` — same layout as `summary_rows/` but with **importance-scaled red** (darker red = higher GNNExplainer edge-mask score) and an extra **"Avg (all strains)"** column showing per-node attribution averaged across all 5 strains:
  - `summary_correct_toxic.pdf`, `summary_correct_nontoxic.pdf`, `summary_incorrect.pdf`
  - `summary_correct_toxic_smiles.csv`, `summary_correct_nontoxic_smiles.csv`, `summary_incorrect_smiles.csv` — SMILES strings in PDF row order, plus `avg_attributions` column (JSON dict mapping atom index → mean edge-mask score across 5 strains)

*Novel-fragment discovery (GNNExplainer path):*

Recurring substructures are obtained as the radius-2/3 circular environments around the model's important atoms. Each is screened against an **extended** alert list (the base alerts plus a few novelty-only SMARTS: any nitro, aromatic azo, poly-halo alkanes/alkenes, sulfonate/sulfate esters) and Tanimoto similarity, then a fragment is called *novel* if it is organic, ≥5 heavy atoms, has a ring or ≥2 heteroatoms, occurs ≥5 times, matches no known/extended alert, and is **statistically enriched** in mutagenic predictions (one-sided binomial test + Benjamini–Hochberg FDR, q<0.05).
- `explainer_discovered_fragments_summary.csv` — every substructure with per-task and positive-prediction counts and matched alerts
- `explainer_novel_fragment_candidates.csv` — the FDR-significant novel substructures (with `pos_frac`, `pval`, `qval`)
- `top_discovered_fragments_grid.pdf` — grid image of the most frequent discovered substructures
- `fragments_known_vs_novel_combined.png` — side-by-side grid of known-alert vs novel substructures

---



## Standalone SHAP Analysis (`shap_analysis_standalone.py`)

A separate, GPU-friendly script computes **grouped KernelSHAP** (true Shapley values, via the `shap`
library). It runs in its **own virtual environment** (it pulls `shap`/numpy 2.x, which conflicts with `mordred` in
the main env) and depends only on `torch` + `torch_geometric`


### 1. Create and activate the conda environment
```bash
conda create -n ames_shap python=3.13
conda activate ames_shap
```

### 2. Install PyTorch
```bash
pip install torch==2.8.0
```

### 3. Install remaining dependencies
```bash
pip install -r requirements_shap.txt
```

### Run

```bash
python shap_analysis_standalone.py \
    --input_file train_sample.yml \
    --checkpoint_file ./checkpoints/final/metrics_77_0.pt \
    --output_dir ./output/SHAP \
    --device cpu \
    --shap_max_mols 3 \
    --shap_nsamples 256 \
```

Outputs (in `<output_dir>/feature_importance_plots_SHAP/`): the SHAP beeswarm violin (`.png`/`.svg`/
`_values.csv`) plus per-strain and overall SHAP bar charts and heatmaps.


Flags: `--shap_max_mols` (molecules per strain), `--shap_nsamples`
(`auto` = 2·M+2048, or an integer), `--shap_chunk` (for GPU batching),
`--device` (`auto`/`cuda`/`cpu`), `--tasks` (e.g. `0,1` for a subset of strains).

---
