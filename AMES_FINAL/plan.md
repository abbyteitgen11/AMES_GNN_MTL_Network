# AMES GNN-MTL Codebase — Code Overview and Changes

## Project Goal

Train a Graph Neural Network Multi-Task Learning (GNN-MTL) model to predict Ames mutagenicity using the ISSSTY dataset. The model predicts mutagenicity across five bacterial strains (TA98, TA100, TA102, TA1535, TA1537) simultaneously using a shared GINEConv backbone with task-specific heads.

---

## Directory Structure

```
AMES_FINAL/
├── run_model.py                  # Consolidated training/evaluation driver (NEW)
├── GNN_explainer_analysis.py     # Consolidated explainer script (NEW)
├── visualize_graphs.py           # Graph database sanity-check visualization (NEW)
├── smiles_to_xyz.py              # SMILES → XYZ conversion (largest-fragment filter added)
├── train_sample.yml              # YAML config for model training (UPDATED paths)
├── optuna/                       # Optuna HPO study .pkl files (NEW directory)
├── checkpoints/                  # Model checkpoints .pt files (NEW directory)
│
├── BuildNN_GNN_MTL_GINEConv.py   # Model class (GINEConv architecture) — DO NOT MODIFY
├── callbacks.py                  # Training callbacks (EarlyStopping, LRScheduler, UserStopping)
├── compute_metrics.py            # Metric computation (Sp, Sn, Acc, F1, H score, etc.)
├── data.py                       # Data loading utilities
├── load_data.py                  # Data loading helpers
├── graph_dataset.py              # PyG GraphDataSet class
├── masked_loss_function.py       # BCE loss ignoring -1 labels with class weights
├── set_seed.py                   # Random seed setting
├── MTLDataset.py                 # Multi-task learning dataset
├── TaskSpecificGNN.py            # Task-specific GNN wrapper for explainability
├── count_model_parameters.py     # Model parameter counting
├── device.py                     # Device selection (CPU/GPU)
│
├── GNN_MTL_GPU.py                # SUPERSEDED by run_model.py (train mode)
├── GNN_MTL_eval.py               # SUPERSEDED by run_model.py (eval mode)
├── GNN_MTL_HP_KF.py              # SUPERSEDED by run_model.py (hp_opt mode)
├── GNN_MTL_HP_KF_seeds.py        # SUPERSEDED by run_model.py (seeds_cfv mode)
├── analyze_crossfold_val.py      # SUPERSEDED by run_model.py (analyze_cfv mode)
├── visualize_optuna.py           # SUPERSEDED by run_model.py (viz_optuna mode)
├── GNN_explainer_analysis_final.py         # SUPERSEDED by GNN_explainer_analysis.py
└── GNN_explainer_analysis_input_features.py # SUPERSEDED by GNN_explainer_analysis.py
```

---

## Model Architecture (`BuildNN_GNN_MTL_GINEConv.py`)

**Constructor signature:**
```python
BuildNN_GNN_MTL(n_gc_layers, n_node_neurons, n_edge_neurons, n_node_features,
                n_edge_features, dropout_GNN, momentum_batch_norm,
                n_s_layers, n_ts_layers, n_shared, n_target,
                dropout_shared, dropout_target, act,
                mode, n_descriptor_inputs=0)
```

**Forward signature:**
```python
forward(x, edge_index, edge_attr, batch, n_node_neurons, n_node_features,
        n_edge_neurons, n_edge_features, n_gc_layers, n_s_layers, n_ts_layers,
        mode, descriptors=None)
```

- GINEConv layers with MLPs, BatchNorm, global_add_pool
- Shared MLP core followed by 5 task-specific heads (one per strain)
- Returns 5 sigmoid outputs

### Three Input Modes

| `mode` | GNN layers | First shared-layer input | `descriptors` arg |
|--------|-----------|--------------------------|-------------------|
| `"gnn"` | Built and used | `global_add_pool(x)` → `[B, n_node_neurons]` | Not used |
| `"descriptor"` | Not built | `descriptors` → `[B, n_descriptor_inputs]` | Required |
| `"combined"` | Built and used | `torch.cat([global_add_pool(x), descriptors], dim=1)` → `[B, n_node_neurons + n_descriptor_inputs]` | Required |

**Data flow for combined mode:**
```
SMILES → GINEConv layers → global_add_pool → [B, n_node_neurons]
                                                          ↘
                                               torch.cat → shared MLP → 5 task heads
                                                          ↗
SMILES → Mordred 2D descriptors → load_descriptor_dict → [B, n_descriptor_inputs]
```

**Descriptor dict lookup** (`load_descriptor_dict` in `run_model.py`): reads descriptor columns from the CSV by column name (not position), mean-imputes NaN per column, returns `{mol_id: np.float32 array}`. The mol_id is extracted from the pkl filename prefix (`int(Path(f).stem.split("_")[0])`).

**New helper `calculate_descriptors.py`**: computes all 2D Mordred descriptors for every SMILES in `data_new_with_split.csv` and writes `data_new_with_split_descriptors.csv` with descriptor columns between `source` and `TA98`.

**`data.py` fix**: replaced positional slice `values[:,3:-7]` with column-name extraction via `_NON_DESC_COLS` set, making it compatible with both `data.csv` and `data_new_with_split_descriptors.csv`.

**Best hyperparameters** (from `train_sample.yml`):
- 3 graph convolutional layers, 78 node neurons, 109 edge neurons
- 2 shared layers [167, 108 neurons], 2 target-specific layers [96, 104 neurons]
- Tanh activation, learning rate 0.00027

---

## `run_model.py` — Consolidated Training Driver

**Usage:**
```bash
python run_model.py --mode <mode> --output_dir <dir> [--input_file <yaml>] [options]
```

### Modes

| Mode | Description |
|------|-------------|
| `train` | Train with fixed HP from YAML; save checkpoints + TensorBoard logs |
| `hp_opt` | Optuna HPO with 5-fold CV; save/resume study .pkl |
| `seeds_cfv` | 5-fold CV across multiple seeds; save per-fold metrics + checkpoints |
| `eval` | Load checkpoint, optimize thresholds on val set, evaluate on test set |
| `top_seeds_eval` | Auto-pick top N seeds by avg val loss; evaluate and average test metrics |
| `analyze_cfv` | Post-hoc analysis of CFV metric CSVs (plots + stats) |
| `viz_optuna` | Visualize Optuna study: history, param importances, val loss |

### Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | One of the modes above |
| `--output_dir` | required | Directory for outputs |
| `--input_file` | required (most modes) | Path to YAML config |
| `--checkpoint_file` | required for eval | Path to .pt checkpoint |
| `--checkpoints_dir` | `AMES_FINAL/checkpoints/` | Where checkpoints are saved/loaded |
| `--optuna_dir` | `AMES_FINAL/optuna/` | Directory for Optuna .pkl files |
| `--optuna_file` | None (auto-dated) | Specific Optuna study file |
| `--metrics_dir` | `output_dir` | Directory containing metrics CSVs for analyze_cfv/top_seeds_eval |
| `--n_trials` | 100 | Number of Optuna trials |
| `--seeds` | `3 7 15 24 42 45 62 77 79 88 90` | Seeds for seeds_cfv mode |
| `--n_top_seeds` | 5 | Number of top seeds for top_seeds_eval |
| `--data_file` | from YAML | Override YAML data_file path |
| `--val_loss_file` | None | Optional .xlsx for validation loss heatmap (analyze_cfv) |
| `--use_thresholds` | False | Optimise per-task decision thresholds on val set before test eval (eval, top_seeds_eval) |
| `--threshold_metric` | `sn` | Metric to maximise when optimising thresholds: `sn`, `sp`, `bal_acc`, `ppv`, `npv`, `mcc`, `f1`, `h` |
| `--tune_consensus_threshold` | False | With `--use_thresholds`: optimise a single shared threshold on the consensus (OR) instead of 5 per-task thresholds |
| `--temperature_scaling` | False | Fit temperature T on val set (minimising NLL) and apply before thresholding (eval, top_seeds_eval) |

### Example commands

```bash
# Train with fixed HP
python run_model.py --mode train --input_file train_sample.yml --output_dir output/train_run1

# Hyperparameter optimization (2 trials, resume existing study)
python run_model.py --mode hp_opt --input_file train_sample.yml --output_dir output/hp_opt \
    --n_trials 2 --optuna_file optuna/study_20260101.pkl

# 5-fold CV across seeds
python run_model.py --mode seeds_cfv --input_file train_sample.yml \
    --output_dir output/seeds_run --seeds "42 45 77"

# Evaluate a checkpoint
python run_model.py --mode eval --input_file train_sample.yml \
    --output_dir output/eval --checkpoint_file checkpoints/metrics_45_1.pt

# Pick top 5 seeds and evaluate
python run_model.py --mode top_seeds_eval --input_file train_sample.yml \
    --output_dir output/top5 --metrics_dir output/seeds_run --n_top_seeds 5

# Analyze cross-fold validation results
python run_model.py --mode analyze_cfv --output_dir output/cfv_plots \
    --metrics_dir output/seeds_run

# Visualize Optuna study
python run_model.py --mode viz_optuna --output_dir output/optuna_plots \
    --optuna_file optuna/study_20260101.pkl
```

---

## `GNN_explainer_analysis.py` — Consolidated Explainer Script

Merged from `GNN_explainer_analysis_final.py` and `GNN_explainer_analysis_input_features.py`.

**Usage:**
```bash
python GNN_explainer_analysis.py \
    --output_dir <dir> \
    --input_file <yaml> \
    --checkpoint_file <path_to_checkpoint.pt> \
    [--data_file <path_to_data.csv>] \
    [--analyze_input_features]
```

### What it does

**Always runs (GNNExplainer analysis):**
- Runs GNNExplainer on each molecule in the test set (all 5 tasks)
- Identifies important atoms (tight 15% edges filter, loose 15% edges + largest connected component)
- Computes overlap between important atoms and structural alert SMARTS patterns
- Builds fragment catalog; identifies novel vs. known-alert fragments
- Generates PDF summary grids (correct toxic, correct non-toxic, incorrect)
- Analyzes per-atom overlap by alert with averaged environment hashing
- Computes toxic overlap heatmap by strain
- Outputs: fragment CSVs, PDF grids, alert overlap CSVs, heatmaps

**With `--analyze_input_features` (Integrated Gradients):**
- Runs Integrated Gradients (50 steps) per molecule per task
- Computes node and edge feature importance (tight 15%, loose 30% edges)
- Aggregates across molecules and tasks
- Saves to `<output_dir>/feature_importance_plots/`:
  - Per-task barplots for node and edge features
  - Overall barplots
  - Heatmaps (tasks × features)

### Structural alerts

35 SMARTS-based alerts covering: acyl halides, nitro groups, aromatic amines, hydrazines, epoxides, quinones, mustards, azo groups, etc.

---

## `visualize_graphs.py` — Graph Database Sanity-Check

Loops through the graph database and renders each molecular graph side-by-side with its RDKit 2D structure, for visual verification of graph construction.

**Usage:**
```bash
python visualize_graphs.py \
    --input_file train_sample.yml \
    --n_graphs 100 \
    --partition test \
    --output_dir ./graph_viz \
    --output_format pdf
```

**Data sources:**
- `GraphDataBase_AMES_NEW/{train,validate,test}/*.pkl` — PyG `MyData` objects
- `data_file` CSV — mol ID → SMILES and toxicity labels
- `GraphDataBase_AMES_NEW/graph_description.yml` — species list (42 elements)

**Graph attributes used:**
- `data.spec_id` — element index per atom (into species list)
- `data.edge_index` — connectivity `[2, num_edges]`
- `data.edge_attr[:,0]` — bond distance in Å
- `data.pos[:,:2]` — x,y projection of 3D coords for 2D layout
- `data.y` — `[1, 5]` toxicity labels

**Layout:** 2 panels per figure — RDKit 2D structure (left) + NetworkX graph (right) with CPK node colors, edge color = bond distance (viridis colorbar), suptitle with mol ID and per-strain labels.

**Key functions:**
- `load_species_list(database_dir)` — reads from `graph_description.yml`
- `collect_pkl_files(database_dir, partition, n_graphs)` — sorted glob, first N
- `draw_structure_2d(smiles, ax, mol_id)` — RDKit `Draw.MolToImage`
- `draw_graph_panel(graph, species_list, show_H, ax)` — NetworkX + matplotlib
- `make_figure(pkl_path, ...)` — assembles 2-panel figure

**Output:** `graphs.pdf` (PDF mode) or `fig_{mol_id}.png` files (PNG mode).

---

## `smiles_to_xyz.py` — SMILES Pre-processing

Converts SMILES strings to XYZ files for subsequent graph construction. Includes a **largest-fragment filter** that discards salts and counter-ions from multi-component SMILES.

**Filter logic** (applied before `Chem.AddHs`, after `Chem.MolFromSmiles`):
```python
frags = Chem.GetMolFrags(mol, asMols=True)
if len(frags) > 1:
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    print(f"  Row {i}: multi-fragment SMILES — keeping largest fragment ...")
```

- Single-atom results are allowed to proceed through the rest of the pipeline.
- Log line printed for every molecule where filtering occurs.
- Example: `COS(=O)(=O)[O-].C[n+]1c2ccccc2nc2ccccc21` → keeps the acridine fragment.

---

## `train_sample.yml` — YAML Training Config

The YAML file controls all model and training hyperparameters. Key fields:

| Field | Description |
|-------|-------------|
| `nGraphConvolutionLayers` | GCL count (3) |
| `nNodeNeurons` / `nEdgeNeurons` | GNN hidden dims (78, 109) |
| `nShared` / `nTarget` | MLP hidden dims |
| `nEpochs` / `nBatch` | Training epochs and batch size |
| `learningRate` | Initial LR (0.00027) |
| `weightedCostFunction` | Enable weighted BCE loss |
| `w1..w5` | Per-strain loss weights |
| `database` | Path to graph database (`GraphDataBase_AMES/`) — keep as-is |
| `data_file` | Path to `AMES_FINAL/data.csv` |
| `loadModel` / `StateDictFileName` | Resume from checkpoint |
| `callbacks` | EarlyStopping, LRScheduler, UserStopping config |

---

## Changes Made

### Bugs fixed

1. **Broken import in `GNN_MTL_HP_KF.py`**: `from BuildNN_GNN_MTL_HP import BuildNN_GNN_MTL` (file doesn't exist) → fixed to use `BuildNN_GNN_MTL_GINEConv`
2. **Broken import in `GNN_MTL_HP_KF_seeds.py`**: `from BuildNN_GNN_MTL import BuildNN_GNN_MTL` → same fix
3. **Incorrect model call in `GNN_MTL_HP_KF.py`**: `model = BuildNN_GNN_MTL(trial, ...)` passed Optuna `trial` as first arg; GINEConv model does not accept it → removed
4. **`val_losses` accumulation bug in `GNN_MTL_HP_KF_seeds.py`**: list not reset between seeds, making avg_val_loss cumulative → fixed with per-seed list
5. **Hardcoded seed headers in `avg_val_losses.csv`**: Fixed to use actual `--seeds` values
6. **Plots saved to current directory in `analyze_crossfold_val.py`**: Fixed to save to `--output_dir`
7. **`verbose` kwarg removed from `ReduceLROnPlateau`** in `callbacks.py` — removed in PyTorch 2.2+
8. **Callbacks monitoring wrong loss**: Both `LRScheduler` and `EarlyStopping` were called with `train_loss` instead of `val_loss`
9. **Early stopping `break` in wrong scope**: `break` was inside `for callback in callbacks` loop, not the epoch loop — fixed with `any(cb.early_stop for cb in callbacks)` check
10. **`eval` mode thresholds discarded**: `crossfit_thresholds_for_consensus` result was computed but never applied to the test set (hardcoded `[0.5]*5` was used instead) — fixed
11. **`eval_consensus_metric` ignored metric parameter**: returned H1 score for any metric except `"bal_acc"` — fixed with proper metric dispatch dict
12. **`metrics_cons.csv` labeled "Strain TA98"** instead of "Consensus" — fixed

### Hardcoded paths removed

| File | Fix |
|------|-----|
| `GNN_MTL_HP_KF.py` optuna study path | `--optuna_dir` flag |
| `GNN_MTL_eval.py` checkpoint path | `--checkpoint_file` flag |
| `analyze_crossfold_val.py` metrics dir | `--metrics_dir` flag |
| `visualize_optuna.py` all study paths | `--optuna_dir` / `--optuna_file` flags |
| `GNN_explainer_analysis_final.py` checkpoint | `--checkpoint_file` flag |
| `GNN_explainer_analysis_final.py` data.csv | `--data_file` flag |
| `GNN_explainer_analysis_input_features.py` checkpoint + data.csv | same flags |
| `train_sample.yml` data_file + StateDictFileName | Updated to AMES_FINAL paths |

### Metrics changes

- All metrics CSVs now include **NPV** (negative predictive value) and **MCC** (Matthews correlation coefficient)
- **"Prec"** column renamed to **"PPV"** (positive predictive value) everywhere
- New column order: `Strain, TP, TN, FP, FN, Sp, Sn, PPV, NPV, Acc, Bal acc, MCC, F1 score, H score`
- `analyze_cfv` mode: removed post-hoc NPV/MCC computation (now written directly by `write_metrics_csv`/`metrics_row`)

### Eval pipeline additions (`eval` and `top_seeds_eval` modes)

- **Temperature scaling** (`--temperature_scaling`): fits scalar T on val set to minimise NLL; applies `sigmoid(logit(p)/T)` before thresholding. Model outputs are probabilities (post-sigmoid), so temperature scaling converts prob→logit→divide by T→sigmoid.
- **Consensus threshold** (`--tune_consensus_threshold`): optimises a single shared threshold via 1D cross-fit grid search instead of 5 per-task coordinate ascent
- **ROC and PR curves** (always in `eval` mode): saved as `roc_curves.png` and `pr_curves.png`; covers each of the 5 strains plus the consensus (using max task probability as consensus score)

### Dead code removed from consolidation

- `analyze_crossfold_val.py` lines 1-132: entire first block was inside a `"""..."""` string (not executed); only the active code (lines 134+) was ported
- `GNN_explainer_analysis_input_features.py` lines 315-472: old GNNExplainer section inside `"""..."""`; only the active IG code was ported

---

## Output Files Reference

### `run_model.py` outputs

| Mode | Output |
|------|--------|
| `train` | `checkpoints/checkpoint_epoch_N.pt`, `tensorboard/` |
| `hp_opt` | `optuna/study_YYYYMMDD.pkl` |
| `seeds_cfv` | `metrics_{seed}_{fold}.csv`, `checkpoints/metrics_{seed}_{fold}.pt`, `avg_val_losses.csv`, `val_losses.csv` |
| `eval` | `metrics.csv`, `metrics_cons.csv`, `misclassified_files.csv`, `model_output_raw.csv`, `roc_curves.png`, `pr_curves.png` |
| `top_seeds_eval` | `top_seeds_all_metrics.csv`, `top_seeds_avg_metrics.csv` |
| `analyze_cfv` | `avg_metrics_by_strain.png`, `metric_distribution_by_strain.png`, `metric_error_bars.png`, `per_seed_variability.png`, `per_fold_variability.png`, optionally `validation_loss_heatmap.png` |
| `viz_optuna` | Interactive plots or saved PNGs |

### `GNN_explainer_analysis.py` outputs

| Output | Description |
|--------|-------------|
| `explainer_discovered_fragments_summary.csv` | Fragment catalog with alert matches |
| `explainer_novel_fragment_candidates.csv` | Fragments not matching known alerts |
| `top_discovered_fragments_grid.pdf` | Top fragment grid image |
| `fragments_known_vs_novel_combined.png` | Grid of alert vs. novel fragments |
| `summary_rows/summary_correct_toxic.pdf` | Per-molecule summary (correct toxic) |
| `summary_rows/summary_correct_nontoxic.pdf` | Per-molecule summary (correct non-toxic) |
| `summary_rows/summary_incorrect.pdf` | Per-molecule summary (incorrect) |
| `alert_averaged_plots_positional/` | Per-alert hashed environment importance plots |
| `alert_instance_grids/` | Per-alert instance grids |
| `toxic_overlap_by_strain_heatmap.pdf` | Heatmap of mean overlap by strain |
| `alert_performance_bars.pdf` | Bar chart of alert detection performance |
| `feature_importance_plots/` | IG feature importance plots (with `--analyze_input_features`) |
