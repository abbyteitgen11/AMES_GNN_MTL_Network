"""
add_negative_examples.py

Find negative examples (ames == 0) in Combined.xlsx that are not already
in data.csv (by canonical SMILES), and merge them into a new CSV.

Usage:
    python add_negative_examples.py \
        --data_csv data.csv \
        --combined_xlsx Combined.xlsx \
        --output_csv data_with_negatives.csv
"""

import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def canonicalize(smiles):
    """Return canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Add novel negative examples from Combined.xlsx to data.csv"
    )
    parser.add_argument(
        "--data_csv",
        default=os.path.join(script_dir, "data.csv"),
        help="Path to existing data CSV (default: data.csv in script directory)",
    )
    parser.add_argument(
        "--combined_xlsx",
        default=os.path.join(script_dir, "Combined.xlsx"),
        help="Path to Combined.xlsx (default: Combined.xlsx in script directory)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(script_dir, "data_with_negatives.csv"),
        help="Output CSV path (default: data_with_negatives.csv in script directory)",
    )
    args = parser.parse_args()

    # ── 1. Load data.csv ──────────────────────────────────────────────────────
    print(f"Loading {args.data_csv} ...")
    raw = pd.read_csv(args.data_csv, low_memory=False)
    print(f"  Rows in data.csv: {len(raw)}")

    keep_cols = ["SMILES RDKit", "TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall"]
    existing = raw[keep_cols].copy()
    existing = existing.rename(columns={"SMILES RDKit": "SMILES"})
    existing["source"] = "isssty"

    # Canonicalize existing SMILES
    canon_smiles = existing["SMILES"].apply(canonicalize)
    n_invalid_existing = canon_smiles.isna().sum()
    if n_invalid_existing > 0:
        print(f"  WARNING: {n_invalid_existing} rows in data.csv have invalid SMILES and will be dropped.")
    existing = existing[canon_smiles.notna()].copy()
    existing["SMILES"] = canon_smiles[canon_smiles.notna()].values
    existing_smiles_set = set(existing["SMILES"])
    print(f"  Rows kept after canonicalization: {len(existing)}")

    # ── 2. Load Combined.xlsx ─────────────────────────────────────────────────
    print(f"\nLoading {args.combined_xlsx} ...")
    combined = pd.read_excel(args.combined_xlsx)
    print(f"  Total rows in Combined.xlsx: {len(combined)}")

    # Filter to negatives
    negatives = combined[combined["ames"] == 0].copy()
    print(f"  Negative examples (ames == 0): {len(negatives)}")

    # Canonicalize
    neg_canon = negatives["smiles"].apply(canonicalize)
    n_invalid_combined = neg_canon.isna().sum()
    if n_invalid_combined > 0:
        print(f"  WARNING: {n_invalid_combined} rows in Combined.xlsx have invalid SMILES and will be skipped.")
    negatives = negatives[neg_canon.notna()].copy()
    negatives["canon_smiles"] = neg_canon[neg_canon.notna()].values

    # Find novel (not in existing set)
    is_novel = ~negatives["canon_smiles"].isin(existing_smiles_set)
    n_duplicates = (~is_novel).sum()
    novel = negatives[is_novel].copy()
    print(f"  Duplicates skipped (already in data.csv): {n_duplicates}")
    print(f"  Novel negatives to add: {len(novel)}")

    # ── 3. Build new rows ─────────────────────────────────────────────────────
    new_rows = pd.DataFrame({
        "SMILES": novel["canon_smiles"].values,
        "source": novel["source"].values,
        "TA98":   0,
        "TA100":  0,
        "TA102":  0,
        "TA1535": 0,
        "TA1537": 0,
        "Overall": 0,
    })

    # ── 4. Concatenate ────────────────────────────────────────────────────────
    output_cols = ["SMILES", "source", "TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall"]
    existing_out = existing[output_cols].copy()
    output_df = pd.concat([existing_out, new_rows], ignore_index=True)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nOutput saved to: {args.output_csv}")
    print(f"  Total rows: {len(output_df)}")
    print(f"  Source breakdown:\n{output_df['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
