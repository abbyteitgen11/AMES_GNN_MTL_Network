"""
build_graphs_new_dataset.py

1. Assign row IDs to data_with_negatives_hansen_eurl.csv (no Id/Partition column yet).
2. Perform a stratified 70/10/20 train/val/test split preserving label proportions
   across all 6 targets (TA98, TA100, TA102, TA1535, TA1537, Overall) including
   the missing-data class (-1).
3. Save split-annotated CSV (data_new_with_split.csv) compatible with load_data.py.
4. Plot label distributions per split for each target.

Usage:
    python build_graphs_new_dataset.py \
        --data_csv data_with_negatives_hansen_eurl.csv \
        --output_csv data_new_with_split.csv \
        --output_dir ./output/split_plots \
        --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


STRAINS = ["TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall"]
LABEL_VALUES = [-1, 0, 1]


# ── Stratified split ──────────────────────────────────────────────────────────

def make_indicator_matrix(df, cols):
    """One-hot encode ternary labels (-1/0/1) → binary indicator matrix (N, 3*len(cols))."""
    parts = []
    for col in cols:
        for val in LABEL_VALUES:
            parts.append((df[col] == val).astype(int).values)
    return np.column_stack(parts)


def stratified_split(df, seed):
    """Return (train_idx, val_idx, test_idx) as numpy arrays of integer positions."""
    Y = make_indicator_matrix(df, STRAINS)
    idx_all = np.arange(len(df))

    # Stage 1: split off test (20%)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trainval_pos, test_pos = next(msss1.split(idx_all, Y))

    # Stage 2: split train+val → train (87.5%) / val (12.5%) → 70% / 10% of total
    Y_tv = Y[trainval_pos]
    idx_tv = np.arange(len(trainval_pos))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_pos_local, val_pos_local = next(msss2.split(idx_tv, Y_tv))

    train_idx = trainval_pos[train_pos_local]
    val_idx   = trainval_pos[val_pos_local]
    test_idx  = test_pos

    return train_idx, val_idx, test_idx


# ── Plotting ──────────────────────────────────────────────────────────────────

SPLIT_COLORS = {"Train": "#4878CF", "Validate": "#6ACC65", "Test": "#D65F5F"}

def plot_distributions(df, output_dir):
    """Grouped bar charts of label distribution per strain per split."""
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(len(LABEL_VALUES))
    width = 0.25
    splits      = ["Train",   "Validate",  "Test"]
    partitions  = ["Train",   "Internal",  "External"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, col in enumerate(STRAINS):
        ax = axes[ax_idx]

        for s_idx, (split_label, partition) in enumerate(zip(splits, partitions)):
            subset = df[df["Partition"] == partition]
            counts = [(subset[col] == v).sum() for v in LABEL_VALUES]
            ax.bar(x + s_idx * width, counts, width,
                   label=split_label, color=SPLIT_COLORS[split_label])

        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("Label  (−1 = missing, 0 = negative, 1 = positive)")
        ax.set_ylabel("Count")
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(v) for v in LABEL_VALUES])
        ax.legend(fontsize=8)

        # Individual plot
        fig_s, ax_s = plt.subplots(figsize=(6, 4))
        for s_idx, (split_label, partition) in enumerate(zip(splits, partitions)):
            subset = df[df["Partition"] == partition]
            counts = [(subset[col] == v).sum() for v in LABEL_VALUES]
            ax_s.bar(x + s_idx * width, counts, width,
                     label=split_label, color=SPLIT_COLORS[split_label])
        ax_s.set_title(f"{col} — Label Distribution by Split")
        ax_s.set_xlabel("Label  (−1 = missing, 0 = negative, 1 = positive)")
        ax_s.set_ylabel("Count")
        ax_s.set_xticks(x + width)
        ax_s.set_xticklabels([str(v) for v in LABEL_VALUES])
        ax_s.legend()
        fig_s.tight_layout()
        fig_s.savefig(os.path.join(output_dir, f"{col}_label_distribution.png"), dpi=150)
        plt.close(fig_s)

    fig.suptitle("Label Distribution per Strain and Split", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "label_distributions_all.png"), dpi=150)
    plt.close(fig)

    # Per-split plots: x = strain, bars = label (-1, 0, 1)
    LABEL_COLORS = {-1: "#AAAAAA", 0: "#6ACC65", 1: "#D65F5F"}
    x_strains = np.arange(len(STRAINS))

    for split_label, partition in zip(splits, partitions):
        subset = df[df["Partition"] == partition]
        fig_p, ax_p = plt.subplots(figsize=(10, 5))
        for l_idx, val in enumerate(LABEL_VALUES):
            counts = [(subset[col] == val).sum() for col in STRAINS]
            ax_p.bar(x_strains + l_idx * width, counts, width,
                     label=str(val), color=LABEL_COLORS[val])
        ax_p.set_title(f"{split_label} Set — Label Distribution by Strain  (n={len(subset)})",
                       fontsize=12, fontweight="bold")
        ax_p.set_xlabel("Strain")
        ax_p.set_ylabel("Count")
        ax_p.set_xticks(x_strains + width)
        ax_p.set_xticklabels(STRAINS)
        ax_p.legend(title="Label  (−1=missing, 0=neg, 1=pos)")
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(output_dir, f"{split_label.lower()}_by_strain.png"), dpi=150)
        plt.close(fig_p)

    print(f"  Plots saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Stratified 70/10/20 split for new AMES dataset + distribution plots"
    )
    parser.add_argument(
        "--data_csv",
        default=os.path.join(script_dir, "data_with_negatives_hansen_eurl.csv"),
        help="Input CSV (default: data_with_negatives_hansen_eurl.csv)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(script_dir, "data_new_with_split.csv"),
        help="Output CSV with Id and Partition columns (default: data_new_with_split.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(script_dir, "output", "split_plots"),
        help="Directory for distribution plots (default: output/split_plots/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.data_csv} ...")
    df = pd.read_csv(args.data_csv)
    print(f"  Rows: {len(df)}")
    df["Id"] = range(1, len(df) + 1)

    # ── Stratified split ──────────────────────────────────────────────────────
    print(f"Performing stratified 70/10/20 split (seed={args.seed}) ...")
    train_idx, val_idx, test_idx = stratified_split(df, args.seed)
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    df["Partition"] = ""
    df.iloc[train_idx, df.columns.get_loc("Partition")] = "Train"
    df.iloc[val_idx,   df.columns.get_loc("Partition")] = "Internal"
    df.iloc[test_idx,  df.columns.get_loc("Partition")] = "External"

    # Label balance summary
    print("\nLabel distribution:")
    for strain in STRAINS:
        print(f"  {strain}:")
        for split_label, partition in [("Train", "Train"), ("Val", "Internal"), ("Test", "External")]:
            subset = df[df["Partition"] == partition]
            counts = {v: (subset[strain] == v).sum() for v in LABEL_VALUES}
            total = len(subset)
            parts = "  ".join(
                f"{v}: {counts[v]} ({100*counts[v]/total:.1f}%)" for v in LABEL_VALUES
            )
            print(f"    {split_label:5s} (n={total}):  {parts}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    strain_cols = [c for c in STRAINS if c != "Overall"]
    out_cols = ["Id", "SMILES", "source"] + strain_cols + ["Overall", "Partition"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.output_csv, index=False)
    print(f"\nSplit CSV saved to: {args.output_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating distribution plots ...")
    plot_distributions(df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
