
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import math
import argparse

def plot_smiles_grid(input_csv, smiles_column="substructure_smiles", n_max=32, mols_per_row=4, out_png="fragments_grid.png"):
    df = pd.read_csv(input_csv)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in {input_csv}. Available: {list(df.columns)}")
    smiles = [s for s in df[smiles_column].astype(str).tolist() if isinstance(s, str) and len(s) > 0]
    smiles = smiles[:n_max]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    legends = [s for s in smiles]
    # Filter Nones (invalid SMILES)
    pairs = [(m,l) for m,l in zip(mols, legends) if m is not None]
    if not pairs:
        raise ValueError("No valid SMILES to draw.")
    mols, legends = zip(*pairs)
    img = Draw.MolsToGridImage(list(mols), molsPerRow=mols_per_row, subImgSize=(250,250), legends=list(legends))
    img.save(out_png)
    print(f"Saved grid image to {out_png}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="CSV file containing a column with SMILES strings")
    p.add_argument("--smiles_column", default="substructure_smiles", help="Name of column with SMILES (default: substructure_smiles)")
    p.add_argument("--n_max", type=int, default=32, help="Max molecules to plot")
    p.add_argument("--mols_per_row", type=int, default=4, help="Molecules per row")
    p.add_argument("--out_png", default="fragments_grid.png", help="Output PNG filename")
    args = p.parse_args()
    plot_smiles_grid(args.input_csv, smiles_column=args.smiles_column, n_max=args.n_max, mols_per_row=args.mols_per_row, out_png=args.out_png)
