"""
Standalone KernelSHAP feature analysis

Run (after creating the venv, see requirements_shap.txt):
    python shap_analysis_standalone.py \
        --input_file <train.yml> \
        --checkpoint_file <model.pt> \
        --output_dir <out_dir> \
        --device auto --shap_max_mols 20 --shap_nsamples auto

GPU: set --device cuda (or auto)
"""

import argparse
import logging
import os
from math import factorial  # noqa: F401  (kept for parity / optional exact fallback)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import yaml
import torch
import shap

from graph_dataset import GraphDataSet
from BuildNN_GNN_MTL_GINEConv import BuildNN_GNN_MTL
from TaskSpecificGNN import TaskSpecificGNN


# ---------------------------------------------------------------------------
# Feature grouping (mirrors the IG analysis groupings)
# ---------------------------------------------------------------------------
NODE_FEATURE_NAMES = [
    "Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Period 6", "Period 7",
    "s block", "p block", "d block", "f block",
    "Alkali metals", "Alkaline earth metals", "Transition metals", "Poor metals", "Metalloids",
    "Nonmetals", "Halogens", "Noble gasses", "Lanthanides", "Actinides",
    "Atomic number", "Atomic radius", "Atomic weight", "Covalent radius", "Density",
    "Pauling electronegativity", "Mass number", "Van der Waals radius",
]
NODE_GROUPS = [
    ("Period", ["Period 1", "Period 2", "Period 3", "Period 4",
                "Period 5", "Period 6", "Period 7"]),
    ("Block", ["s block", "p block", "d block", "f block"]),
    ("Element group", ["Alkali metals", "Alkaline earth metals", "Transition metals", "Poor metals",
                       "Metalloids", "Nonmetals", "Halogens", "Noble gasses",
                       "Lanthanides", "Actinides"]),
]


def build_group_column_maps(node_feature_names, node_groups, n_dist_feats,
                            edge_feature_names, n_node_features, n_edge_features):
    """Map raw feature columns to grouped SHAP players. Node players first, then edge players."""
    group_names, group_is_edge = [], []
    node_col_to_group = np.full(n_node_features, -1, dtype=int)
    edge_col_to_group = np.full(n_edge_features, -1, dtype=int)
    name_to_col = {n: i for i, n in enumerate(node_feature_names) if i < n_node_features}
    grouped_members = set()

    for gname, members in node_groups:
        gidx = len(group_names)
        group_names.append(gname); group_is_edge.append(False)
        for m in members:
            grouped_members.add(m)
            if m in name_to_col:
                node_col_to_group[name_to_col[m]] = gidx

    for name in node_feature_names:
        if name in grouped_members or name not in name_to_col:
            continue
        gidx = len(group_names)
        group_names.append(name); group_is_edge.append(False)
        node_col_to_group[name_to_col[name]] = gidx

    n_node_groups = len(group_names)

    dist_idx = len(group_names)
    group_names.append(edge_feature_names[0]); group_is_edge.append(True)  # "Distance"
    for c in range(min(n_dist_feats, n_edge_features)):
        edge_col_to_group[c] = dist_idx
    for offset, name in enumerate(edge_feature_names[1:]):
        col = n_dist_feats + offset
        if col >= n_edge_features:
            break
        gidx = len(group_names)
        group_names.append(name); group_is_edge.append(True)
        edge_col_to_group[col] = gidx

    return group_names, np.array(group_is_edge), node_col_to_group, edge_col_to_group, n_node_groups


def _masked_forward(task_model, x, edge_index, edge_attr, Z,
                    node_col_to_group, edge_col_to_group, device, chunk=256):
    """Z: (n, M) of 0/1. Zero the columns of players whose bit is 0; batched per-graph forward."""
    n = Z.shape[0]
    N, Fn = x.shape
    E = edge_index.shape[1]
    Fe = edge_attr.shape[1]
    valid_n = node_col_to_group >= 0
    valid_e = edge_col_to_group >= 0
    gn = node_col_to_group[valid_n]
    ge = edge_col_to_group[valid_e]

    out_all = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        Zc = Z[start:start + chunk]
        b = Zc.shape[0]
        node_mask = np.ones((b, Fn), dtype=np.float32); node_mask[:, valid_n] = Zc[:, gn]
        edge_mask = np.ones((b, Fe), dtype=np.float32); edge_mask[:, valid_e] = Zc[:, ge]
        node_mask_t = torch.from_numpy(node_mask).to(device)
        edge_mask_t = torch.from_numpy(edge_mask).to(device)

        x_big = (x.unsqueeze(0) * node_mask_t.unsqueeze(1)).reshape(b * N, Fn)
        edge_attr_big = (edge_attr.unsqueeze(0) * edge_mask_t.unsqueeze(1)).reshape(b * E, Fe)
        offsets = (torch.arange(b, device=device) * N).repeat_interleave(E)
        edge_index_big = edge_index.repeat(1, b) + offsets.unsqueeze(0)
        batch_big = torch.arange(b, device=device).repeat_interleave(N)

        with torch.no_grad():
            out = task_model(x=x_big, edge_index=edge_index_big,
                             edge_attr=edge_attr_big, batch=batch_big)
        out = out.reshape(-1)
        if out.numel() != b:
            raise RuntimeError(
                f"Batched forward returned {out.numel()} outputs for {b} graphs; the model does not "
                "appear to pool by `batch`.")
        out_all[start:start + b] = out.detach().cpu().numpy()
    return out_all


def grouped_feature_values(x_np, edge_np, node_col_to_group, edge_col_to_group,
                           group_is_edge, n_dist_feats, mu, M):
    """Per-molecule grouped feature value for the violin colormap (Distance decoded to Angstrom)."""
    vals = np.zeros(M, dtype=float)
    for g in range(M):
        if group_is_edge[g]:
            cols = np.where(edge_col_to_group == g)[0]
            if cols.size == 0:
                continue
            if cols.size == n_dist_feats and n_dist_feats > 1:
                rbf_mean = edge_np[:, cols].mean(axis=0)
                denom = rbf_mean.sum()
                vals[g] = float(rbf_mean @ mu / denom) if denom > 0 else 0.0
            else:
                vals[g] = float(edge_np[:, cols].mean())
        else:
            cols = np.where(node_col_to_group == g)[0]
            if cols.size > 1:
                # One-hot family (Period/Block/Element group): mean active category index across atoms
                # (columns are in ordinal order). The plain mean of indicators is a constant 1/group_size
                # for every molecule, so it carries no color information.
                sub = x_np[:, cols]
                w = sub.sum(axis=1)
                w = np.where(w == 0, 1.0, w)
                vals[g] = float(((sub @ np.arange(cols.size)) / w).mean())
            elif cols.size == 1:
                vals[g] = float(x_np[:, cols].mean())
    return vals


def shap_for_molecule(task_model, data, node_col_to_group, edge_col_to_group, group_is_edge,
                      n_dist_feats, mu, M, device, chunk, nsamples, log_efficiency=False):
    """KernelSHAP (shap library) over the M grouped players for one molecule."""
    if data.edge_index.shape[1] == 0:
        return None
    x = data.x.detach().float().to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.detach().float().to(device)

    def predict_fn(Z):
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        return _masked_forward(task_model, x, edge_index, edge_attr, Z,
                               node_col_to_group, edge_col_to_group, device, chunk)

    explainer = shap.KernelExplainer(predict_fn, np.zeros((1, M)))
    phi = explainer.shap_values(np.ones((1, M)), nsamples=nsamples, l1_reg=0, silent=True)
    phi = np.asarray(phi)
    if phi.ndim == 2:
        phi = phi[0]
    phi = phi.reshape(-1)

    feat_vals = grouped_feature_values(
        x.detach().cpu().numpy(), edge_attr.detach().cpu().numpy(),
        node_col_to_group, edge_col_to_group, group_is_edge, n_dist_feats, mu, M)

    if log_efficiency:
        f_inst = float(predict_fn(np.ones((1, M)))[0])
        logging.info("[SHAP][efficiency] f(instance)=%.4f  base+sum(phi)=%.4f",
                     f_inst, float(explainer.expected_value) + phi.sum())
    return phi, feat_vals


# ---------------------------------------------------------------------------
# Plotting (self-contained copies, kept visually identical to the IG figures)
# ---------------------------------------------------------------------------
def plot_shap_violin(node_matrix, node_feature_names, edge_matrix, edge_feature_names,
                     node_feat_values, edge_feat_values, node_groups, title, filename, plot_dir):
    """SHAP beeswarm/violin: x = signed SHAP value; color = input feature value (blue low, red high)."""
    from scipy.stats import gaussian_kde

    grouped_node_names, grouped_node_cols, grouped_node_feat_cols = [], [], []
    grouped_indices = set()
    for group_name, group_feats in node_groups:
        idxs = [i for i, n in enumerate(node_feature_names) if n in set(group_feats)]
        if idxs:
            grouped_node_cols.append(node_matrix[:, idxs].mean(axis=1))
            grouped_node_feat_cols.append(node_feat_values[:, idxs].mean(axis=1))
            grouped_node_names.append(group_name)
            grouped_indices.update(idxs)
    for i, name in enumerate(node_feature_names):
        if i not in grouped_indices:
            grouped_node_cols.append(node_matrix[:, i])
            grouped_node_feat_cols.append(node_feat_values[:, i])
            grouped_node_names.append(name)

    node_processed = np.column_stack(grouped_node_cols)
    node_feat_processed = np.column_stack(grouped_node_feat_cols)
    edge_names = list(edge_feature_names)
    feat_type_map = {n: "Node" for n in grouped_node_names}
    feat_type_map.update({n: "Edge" for n in edge_names})

    all_feat_names = grouped_node_names + edge_names
    all_shap = np.hstack([node_processed, edge_matrix])
    all_feat_vals = np.hstack([node_feat_processed, edge_feat_values])

    mean_abs = np.abs(all_shap).mean(axis=0)
    sort_order = np.argsort(-mean_abs)
    sorted_names = [all_feat_names[i] for i in sort_order]
    sorted_shap = all_shap[:, sort_order]
    sorted_feat_vals = all_feat_vals[:, sort_order]

    # Robust 5-95 percentile clip for color (matches shap.summary_plot); min-max is dominated by the
    # heavy right-skew of molecular feature values and washes the bulk to one color.
    feat_lo = np.nanpercentile(sorted_feat_vals, 5, axis=0, keepdims=True)
    feat_hi = np.nanpercentile(sorted_feat_vals, 95, axis=0, keepdims=True)
    feat_range = np.where(feat_hi - feat_lo == 0, 1.0, feat_hi - feat_lo)
    norm_feat_vals = np.clip((sorted_feat_vals - feat_lo) / feat_range, 0.0, 1.0)

    n_feats = len(sorted_names)
    _, ax = plt.subplots(figsize=(10, max(6, n_feats * 0.5)))
    cmap = plt.cm.coolwarm
    rng = np.random.default_rng(seed=42)

    for feat_idx in range(n_feats):
        vals = sorted_shap[:, feat_idx]
        fv_norm = norm_feat_vals[:, feat_idx]
        y_center = feat_idx
        if vals.std() > 1e-10 and len(vals) > 2:
            try:
                kde = gaussian_kde(vals, bw_method='scott')
                x_range = np.linspace(vals.min(), vals.max(), 200)
                kde_density = kde(x_range); kde_density = kde_density / kde_density.max()
                ax.fill_between(x_range, y_center - 0.4 * kde_density, y_center + 0.4 * kde_density,
                                color="lightgray", alpha=0.6, zorder=1)
                kde_at_pts = kde(vals); kde_at_pts = kde_at_pts / kde_at_pts.max()
                jitter = rng.uniform(-1, 1, size=len(vals)) * 0.38 * kde_at_pts
            except Exception:
                jitter = rng.uniform(-0.35, 0.35, size=len(vals))
        else:
            jitter = rng.uniform(-0.35, 0.35, size=len(vals))
        marker = "o" if feat_type_map[sorted_names[feat_idx]] == "Node" else "D"
        ax.scatter(vals, y_center + jitter, c=fv_norm, cmap=cmap, vmin=0, vmax=1,
                   alpha=0.7, s=10, linewidths=0, marker=marker, zorder=3)

    ax.set_yticks(range(n_feats)); ax.set_yticklabels(sorted_names)
    ax.invert_yaxis(); ax.axvline(0, color="darkgray", linewidth=0.8, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)); sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Feature value (normalized)", shrink=0.4, pad=0.01)
    ax.legend(handles=[
        Line2D([], [], color="gray", marker="o", linestyle="None", markersize=6, label="Node feature"),
        Line2D([], [], color="gray", marker="D", linestyle="None", markersize=6, label="Edge feature"),
    ], loc="lower right")
    ax.set_xlabel("SHAP value"); ax.set_ylabel(""); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.savefig(os.path.join(plot_dir, os.path.splitext(filename)[0] + ".svg"))
    plt.close()

    records = []
    for feat_idx in range(n_feats):
        feat_name = sorted_names[feat_idx]
        for sv, fvv in zip(sorted_shap[:, feat_idx], sorted_feat_vals[:, feat_idx]):
            records.append({"Feature": feat_name, "Type": feat_type_map[feat_name],
                            "SHAP Value": sv, "Feature Value": fvv})
    pd.DataFrame(records).to_csv(
        os.path.join(plot_dir, os.path.splitext(filename)[0] + "_values.csv"), index=False)


def plot_task_bars(importances_dict, feature_names, title_prefix, filename_prefix, plot_dir):
    for task_id, values in importances_dict.items():
        if not isinstance(values, np.ndarray):
            continue
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), feature_names, rotation=60, ha="right")
        plt.ylabel("SHAP value"); plt.title(f"{title_prefix} — Task {task_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{filename_prefix}_task_{task_id}.png"), dpi=300)
        plt.close()


def plot_overall_bars(values, feature_names, title, filename, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), feature_names, rotation=60, ha="right")
    plt.ylabel("SHAP value"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


def plot_heatmap(importances_dict, feature_names, title, filename, plot_dir):
    task_ids = sorted(importances_dict.keys())
    matrix = [importances_dict[t] if isinstance(importances_dict[t], np.ndarray)
              else np.zeros(len(feature_names)) for t in task_ids]
    matrix = np.array(matrix)
    if matrix.size == 0:
        return
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix, annot=False, cmap="viridis",
                xticklabels=feature_names, yticklabels=[f"Task {t}" for t in task_ids])
    plt.xticks(rotation=60, ha="right"); plt.xlabel("Features"); plt.ylabel("Tasks"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def replot_violin_from_csv(csv_path, output_dir):
    """Re-render the SHAP violin from an existing `..._values.csv` with the fixed (percentile) color
    normalization, without recomputing SHAP. Rows are molecule-ordered and consistent across features;
    node/edge split comes from the 'Type' column. (One-hot group rows stay flat here, since their
    corrected feature value isn't stored in the CSV — a full re-run picks that up.)"""
    df = pd.read_csv(csv_path)
    node_feats = [f for f, t in df.drop_duplicates("Feature")[["Feature", "Type"]].values if t == "Node"]
    edge_feats = [f for f, t in df.drop_duplicates("Feature")[["Feature", "Type"]].values if t == "Edge"]

    def stack(feats, col):
        cols = []
        n = None
        for f in feats:
            arr = df.loc[df["Feature"] == f, col].to_numpy()
            if n is None:
                n = len(arr)
            elif len(arr) != n:
                raise ValueError(f"Feature '{f}' has {len(arr)} rows, expected {n}; CSV is inconsistent.")
            cols.append(arr)
        return np.column_stack(cols) if cols else np.empty((0, 0))

    node_shap, edge_shap = stack(node_feats, "SHAP Value"), stack(edge_feats, "SHAP Value")
    node_fv, edge_fv = stack(node_feats, "Feature Value"), stack(edge_feats, "Feature Value")
    os.makedirs(output_dir, exist_ok=True)
    plot_shap_violin(node_shap, node_feats, edge_shap, edge_feats, node_fv, edge_fv, [],
                     "Overall Feature Importance — KernelSHAP (per molecule)",
                     "overall_feature_importance_violin_SHAP.png", output_dir)
    logging.info("[SHAP] re-plotted violin from %s into %s", csv_path, output_dir)


def get_args():
    p = argparse.ArgumentParser(description="Standalone grouped-KernelSHAP analysis for the AMES GNN.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--replot_from_csv", default=None,
                   help="Re-render the violin from an existing ..._values.csv with the fixed color "
                        "normalization (no model/SHAP recompute). When set, --input_file/--checkpoint_file "
                        "are ignored.")
    p.add_argument("--input_file", help="Training/eval YAML config.")
    p.add_argument("--checkpoint_file", help="Path to the .pt model checkpoint.")
    p.add_argument("--data_file", default=None, help="Optional; overrides data_file in the YAML.")
    p.add_argument("--shap_max_mols", type=int, default=20,
                   help="Max test molecules per strain head (SHAP is expensive; default 20).")
    p.add_argument("--shap_chunk", type=int, default=256,
                   help="Coalitions per batched GNN forward pass.")
    p.add_argument("--shap_nsamples", default="auto",
                   help="KernelSHAP nsamples: 'auto' (2*M+2048) or an integer.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--tasks", default=None,
                   help="Comma-separated strain-head indices 0-4 (default: all 5).")
    return p.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.replot_from_csv:
        replot_violin_from_csv(args.replot_from_csv, args.output_dir)
        return

    if not args.input_file or not args.checkpoint_file:
        raise SystemExit("--input_file and --checkpoint_file are required unless --replot_from_csv is set.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logging.info("Using device: %s", device)

    nsamples = "auto" if str(args.shap_nsamples) == "auto" else int(args.shap_nsamples)
    tasks = [int(t) for t in args.tasks.split(",")] if args.tasks else list(range(5))

    with open(args.input_file) as f:
        input_data = yaml.load(f, Loader=yaml.Loader)
    database_path = input_data.get("database", "./GraphDataBase_AMES")
    with open(os.path.join(database_path, "graph_description.yml")) as f:
        database_data = yaml.load(f, Loader=yaml.Loader)

    # Model / graph parameters (mirrors GNN_explainer_analysis.main())
    n_graph_convolution_layers = input_data.get("nGraphConvolutionLayers", 0)
    n_node_neurons = input_data.get("nNodeNeurons", None)
    n_edge_neurons = input_data.get("nEdgeNeurons", None)
    dropout_GNN = input_data.get("dropoutGNN", None)
    momentum_batch_norm = input_data.get("momentumBatchNorm", None)
    n_shared_layers = input_data.get("nSharedLayers", 4)
    n_target_specific_layers = input_data.get("nTargetSpecificLayers", 2)
    n_shared = input_data.get("nShared", None)
    n_target = input_data.get("nTarget", None)
    dropout_shared = input_data.get("dropoutShared", None)
    dropout_target = input_data.get("dropoutTarget", None)
    activation = input_data.get("ActivationFunction", "ReLU")
    input_mode = input_data.get("inputMode", None)
    if input_mode is None:
        input_mode = "descriptor" if input_data.get("useMolecularDescriptors", False) else "gnn"

    n_node_features = database_data.get("nNodeFeatures")
    bond_angle_features = database_data.get("BondAngleFeatures", True)
    dihedral_angle_features = database_data.get("DihedralAngleFeatures", True)
    n_dist_feats = database_data.get("nDistanceFeatures", 1)
    n_edge_features = n_dist_feats + (1 if bond_angle_features else 0) + (1 if dihedral_angle_features else 0)

    edge_feature_names = (["Distance"]
                          + (["Bond angle"] if bond_angle_features else [])
                          + (["Dihedral angle"] if dihedral_angle_features else []))

    rbf = database_data.get("RBFParameters", {}) or {}
    r_min = float(rbf.get("r_min", 0.0)); r_max = float(rbf.get("r_max", 5.0))
    mu = np.linspace(r_min, r_max, n_dist_feats) if n_dist_feats > 1 else np.array([0.0])

    testDataset = GraphDataSet(os.path.join(database_path, "test"))

    model = BuildNN_GNN_MTL(n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features,
                            n_edge_features, dropout_GNN, momentum_batch_norm, n_shared_layers,
                            n_target_specific_layers, n_shared, n_target, dropout_shared, dropout_target,
                            activation, input_mode, 0)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval(); model.to(device)

    (group_names, group_is_edge, node_col_to_group,
     edge_col_to_group, n_node_groups) = build_group_column_maps(
        NODE_FEATURE_NAMES, NODE_GROUPS, n_dist_feats, edge_feature_names,
        n_node_features, n_edge_features)
    M = len(group_names)
    node_group_names, edge_group_names = group_names[:n_node_groups], group_names[n_node_groups:]
    logging.info("[SHAP] %d grouped players (%d node, %d edge); KernelSHAP up to %d molecules/strain.",
                 M, n_node_groups, M - n_node_groups, args.shap_max_mols)

    shap_rows, featval_rows = [], []
    shap_node_importance, shap_edge_importance = {}, {}
    model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                  n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, input_mode)

    for task_id in tasks:
        task_model = TaskSpecificGNN(model, task_idx=task_id, model_args=model_args)
        task_model.eval(); task_model.to(device)
        task_phis = []
        n_done = 0
        for data in testDataset:
            if n_done >= args.shap_max_mols:
                break
            data = data.to(device)
            res = shap_for_molecule(task_model, data, node_col_to_group, edge_col_to_group,
                                    group_is_edge, n_dist_feats, mu, M, device, args.shap_chunk,
                                    nsamples, log_efficiency=(n_done == 0))
            if res is None:
                continue
            phi, fvals = res
            shap_rows.append(phi); featval_rows.append(fvals); task_phis.append(phi)
            n_done += 1
        if task_phis:
            task_mean = np.mean(np.vstack(task_phis), axis=0)
            shap_node_importance[task_id] = task_mean[:n_node_groups]
            shap_edge_importance[task_id] = task_mean[n_node_groups:]
        logging.info("[SHAP] strain head %d: %d molecules explained.", task_id, n_done)

    if not shap_rows:
        logging.warning("No molecules were explained (no edges / empty dataset?). Nothing written.")
        return

    plot_dir = os.path.join(args.output_dir, "feature_importance_plots_SHAP")
    os.makedirs(plot_dir, exist_ok=True)

    shap_all = np.vstack(shap_rows)
    featval_all = np.vstack(featval_rows)
    node_shap, edge_shap = shap_all[:, :n_node_groups], shap_all[:, n_node_groups:]
    node_fv, edge_fv = featval_all[:, :n_node_groups], featval_all[:, n_node_groups:]

    plot_shap_violin(node_shap, node_group_names, edge_shap, edge_group_names,
                     node_fv, edge_fv, [],
                     "Overall Feature Importance — KernelSHAP (per molecule)",
                     "overall_feature_importance_violin_SHAP.png", plot_dir)

    plot_task_bars(shap_node_importance, node_group_names, "Node Feature SHAP", "node_feature_shap", plot_dir)
    plot_task_bars(shap_edge_importance, edge_group_names, "Edge Feature SHAP", "edge_feature_shap", plot_dir)
    if shap_node_importance:
        overall_node = np.mean(np.vstack(list(shap_node_importance.values())), axis=0)
        overall_edge = np.mean(np.vstack(list(shap_edge_importance.values())), axis=0)
        plot_overall_bars(overall_node, node_group_names, "Overall Node Feature SHAP",
                          "overall_node_feature_shap.png", plot_dir)
        plot_overall_bars(overall_edge, edge_group_names, "Overall Edge Feature SHAP",
                          "overall_edge_feature_shap.png", plot_dir)
    plot_heatmap(shap_node_importance, node_group_names, "Node Feature SHAP per Strain",
                 "node_feature_shap_heatmap.png", plot_dir)
    plot_heatmap(shap_edge_importance, edge_group_names, "Edge Feature SHAP per Strain",
                 "edge_feature_shap_heatmap.png", plot_dir)
    logging.info("[SHAP] wrote figures to %s (%d molecule explanations).", plot_dir, shap_all.shape[0])


if __name__ == "__main__":
    main()
