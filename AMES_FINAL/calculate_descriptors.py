"""
calculate_descriptors.py

Compute Mordred 2D molecular descriptors for every SMILES in an input CSV
and save a new CSV with descriptor columns inserted between 'source' and 'TA98'.

The output can be used directly with run_model.py in 'descriptor' or 'combined' modes.

Usage:
    python calculate_descriptors.py \
        --input_csv data_new_with_split.csv \
        --output_csv data_new_with_split_descriptors.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    from mordred import Calculator, descriptors as mordred_descriptors
except ImportError:
    raise ImportError(
        "mordred is required. Install with: pip install mordred"
    )


META_COLS  = ["Id", "SMILES", "source"]
LABEL_COLS = ["TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall", "Partition"]


def compute_descriptors(df, smiles_col="SMILES"):
    """Compute all Mordred 2D descriptors for each row. Returns a DataFrame."""
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    desc_names = [str(d) for d in calc.descriptors]

    rows = []
    n = len(df)
    for i, smi in enumerate(df[smiles_col], start=1):
        if i % 500 == 0 or i == n:
            print(f"  {i}/{n} molecules processed ...")

        if not isinstance(smi, str) or smi.strip() == "":
            rows.append({k: np.nan for k in desc_names})
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({k: np.nan for k in desc_names})
            continue

        # Keep only largest fragment (discard salts / counter-ions)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
            print(f"  Row {i}: multi-fragment SMILES — keeping largest fragment "
                  f"({mol.GetNumAtoms()} heavy atoms), discarded {len(frags)-1} other(s)")

        result = calc(mol)
        row = {}
        for name, val in zip(desc_names, result):
            try:
                row[name] = float(val)
            except (TypeError, ValueError):
                row[name] = np.nan
        rows.append(row)

    return pd.DataFrame(rows, index=df.index)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Compute Mordred 2D descriptors and append to a split CSV"
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join(script_dir, "data_new_with_split.csv"),
        help="Input CSV with SMILES, labels, and Partition (default: data_new_with_split.csv)",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(script_dir, "data_new_with_split_descriptors.csv"),
        help="Output CSV with descriptor columns added (default: data_new_with_split_descriptors.csv)",
    )
    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Name of the SMILES column (default: SMILES)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)
    print(f"  Rows: {len(df)}")

    # Validate required columns
    missing = [c for c in META_COLS + LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    print("Computing Mordred 2D descriptors (this may take several minutes) ...")
    desc_df = compute_descriptors(df, smiles_col=args.smiles_col)

    # Report NaN statistics
    n_failed = (desc_df.isna().all(axis=1)).sum()
    if n_failed:
        print(f"  WARNING: {n_failed} molecule(s) produced all-NaN descriptor rows "
              f"(invalid or missing SMILES).")

    print(f"  Descriptor columns computed: {len(desc_df.columns)}")

    # Assemble output: meta | descriptors | labels
    output_cols_present = [c for c in META_COLS if c in df.columns]
    label_cols_present  = [c for c in LABEL_COLS if c in df.columns]
    output = pd.concat([df[output_cols_present], desc_df, df[label_cols_present]], axis=1)

    output.to_csv(args.output_csv, index=False)
    print(f"\nSaved to: {args.output_csv}")
    print(f"  Shape: {output.shape}")
    print(f"  Columns: {list(output.columns[:4])} ... {list(output.columns[-4:])}")
    print("Done.")


if __name__ == "__main__":
    main()
