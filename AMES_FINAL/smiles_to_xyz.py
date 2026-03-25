"""
smiles_to_xyz.py

Convert SMILES strings from a CSV to 3D XYZ files using RDKit conformer generation.

Naming convention: {row}_ames_mutagenicity_data_{row}.xyz  (1-indexed row number)
XYZ format matches the existing DataBase_AMES/FILES_XYZ/ collection.

Usage:
    python smiles_to_xyz.py \
        --input_csv data_with_negatives.csv \
        --smiles_col SMILES \
        --output_dir ./FILES_XYZ_new
"""

import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")


def embed_mol(mol):
    """Generate 3D conformer. Returns mol with conformer, or None on failure."""
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Fallback: random coordinates
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        return None
    return mol


def optimize_mol(mol, row_idx):
    """Optimize conformer geometry. Returns (mol, note_string)."""
    result = AllChem.MMFFOptimizeMolecule(mol)
    if result == 0:
        return mol, ""
    if result == 1:
        # MMFF did not converge — coordinates still usable
        return mol, " (MMFF did not converge)"
    # MMFF failed entirely — try UFF
    result_uff = AllChem.UFFOptimizeMolecule(mol)
    if result_uff in (0, 1):
        note = " (UFF used)" if result_uff == 0 else " (UFF did not converge)"
        return mol, note
    return mol, " (optimization failed, using unoptimized coords)"


def write_xyz(path, row_idx, mol):
    """Write XYZ file matching DataBase_AMES format."""
    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    n_atoms = mol.GetNumAtoms()
    comment = f"{row_idx}_ames_mutagenicity_data"

    with open(path, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{comment}\n")
        for atom in atoms:
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{sym:1s}{pos.x:17.5f}{pos.y:15.5f}{pos.z:15.5f}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Convert SMILES to XYZ files using RDKit 3D conformer generation"
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.join(script_dir, "data_with_negatives.csv"),
        help="Input CSV with SMILES (default: data_with_negatives.csv)",
    )
    parser.add_argument(
        "--smiles_col",
        default="SMILES",
        help="Name of the SMILES column (default: SMILES)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(script_dir, "FILES_XYZ_new"),
        help="Output directory for XYZ files (default: FILES_XYZ_new/)",
    )
    args = parser.parse_args()

    # Load CSV
    print(f"Loading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"  Rows: {len(df)}")

    os.makedirs(args.output_dir, exist_ok=True)

    n_success = 0
    n_skipped = 0

    for i, smiles in enumerate(df[args.smiles_col], start=1):
        row_label = f"{i}_ames_mutagenicity_data_{i}"
        out_path = os.path.join(args.output_dir, f"{row_label}.xyz")

        # Parse SMILES
        if not isinstance(smiles, str) or smiles.strip() == "":
            print(f"  Row {i}: empty/missing SMILES — skipped")
            n_skipped += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  Row {i}: invalid SMILES '{smiles}' — skipped")
            n_skipped += 1
            continue

        # Add explicit hydrogens and generate 3D conformer
        mol = Chem.AddHs(mol)
        mol = embed_mol(mol)
        if mol is None:
            print(f"  Row {i}: 3D embedding failed for '{smiles}' — skipped")
            n_skipped += 1
            continue

        mol, note = optimize_mol(mol, i)
        if note:
            print(f"  Row {i}: {note.strip()}")

        write_xyz(out_path, i, mol)
        n_success += 1

        if i % 500 == 0:
            print(f"  Processed {i}/{len(df)} ...")

    print(f"\nDone.")
    print(f"  Total rows:  {len(df)}")
    print(f"  Succeeded:   {n_success}")
    print(f"  Skipped:     {n_skipped}")
    print(f"  Output dir:  {args.output_dir}")


if __name__ == "__main__":
    main()
