

import argparse
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

def load_alerts(alerts_csv):
    df = pd.read_csv(alerts_csv)
    df = df.dropna(subset=["SMARTS","Class"])
    alerts = []
    for _, row in df.iterrows():
        smarts = str(row["SMARTS"]).strip()
        name = str(row.get("Name", smarts)).strip()
        klass = str(row["Class"]).strip().lower()
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        alerts.append((name, patt, klass))
    return alerts

def compute_score_column(df, sort_by):
    if sort_by and sort_by in df.columns:
        return sort_by
    if "total_pred" in df.columns:
        return "total_pred"
    if set(["pred_pos","pred_neg"]).issubset(df.columns):
        df["_score_tmp"] = df["pred_pos"].fillna(0).astype(float) + df["pred_neg"].fillna(0).astype(float)
        return "_score_tmp"
    df["_score_tmp"] = range(len(df), 0, -1)
    return "_score_tmp"

def collect_highlights(mol, alerts):
    # Return atom/bond highlight info and matched names per class.
    per_class_atoms = {"toxic": set(), "nontoxic": set()}
    per_class_bonds = {"toxic": set(), "nontoxic": set()}
    matched_names = set()

    for name, patt, klass in alerts:
        if klass not in per_class_atoms:
            continue
        matches = mol.GetSubstructMatches(patt, uniquify=True)
        if not matches:
            continue
        matched_names.add(f"{name} ({klass})")
        for match in matches:
            for a in match:
                per_class_atoms[klass].add(a)
            amap = {qi: ti for qi, ti in enumerate(match)}
            for qb in patt.GetBonds():
                qa = qb.GetBeginAtomIdx()
                qb_idx = qb.GetEndAtomIdx()
                ta = amap.get(qa, None)
                tb = amap.get(qb_idx, None)
                if ta is None or tb is None:
                    continue
                b = mol.GetBondBetweenAtoms(ta, tb)
                if b is not None:
                    per_class_bonds[klass].add(b.GetIdx())

    return per_class_atoms, per_class_bonds, sorted(matched_names)

def grid_highlight_topk(input_csv, smiles_column, alerts_csv, top_k=10, sort_by=None, mols_per_row=5, out_png="fragments_highlighted_topk_grid.png"):
    df = pd.read_csv(input_csv)
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in {input_csv}. Available: {list(df.columns)}")

    score_col = compute_score_column(df, sort_by)
    df_sorted = df.sort_values(by=score_col, ascending=False).head(top_k).copy()

    alerts = load_alerts(alerts_csv)
    if not alerts:
        raise ValueError("No valid SMARTS loaded from alerts CSV.")

    COLOR_TOXIC = (0.9, 0.0, 0.0)     # red
    COLOR_NONT = (0.0, 0.2, 0.9)      # blue

    mols, legends = [], []
    hl_atom_lists, hl_bond_lists = [], []
    hl_atom_colors, hl_bond_colors = [], []

    for _, row in df_sorted.iterrows():
        smi = str(row[smiles_column]).strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        Chem.rdDepictor.Compute2DCoords(mol)

        per_class_atoms, per_class_bonds, matched_names = collect_highlights(mol, alerts)

        atoms_all = sorted(per_class_atoms["toxic"].union(per_class_atoms["nontoxic"]))
        bonds_all = sorted(per_class_bonds["toxic"].union(per_class_bonds["nontoxic"]))

        atom_cols = {}
        for a in per_class_atoms["toxic"]:
            atom_cols[a] = COLOR_TOXIC
        for a in per_class_atoms["nontoxic"]:
            if a in atom_cols:
                r1,g1,b1 = atom_cols[a]; r2,g2,b2 = COLOR_NONT
                atom_cols[a] = ((r1+r2)/2.0, (g1+g2)/2.0, (b1+b2)/2.0)
            else:
                atom_cols[a] = COLOR_NONT

        bond_cols = {}
        for b in per_class_bonds["toxic"]:
            bond_cols[b] = COLOR_TOXIC
        for b in per_class_bonds["nontoxic"]:
            if b in bond_cols:
                r1,g1,b1 = bond_cols[b]; r2,g2,b2 = COLOR_NONT
                bond_cols[b] = ((r1+r2)/2.0, (g1+g2)/2.0, (b1+b2)/2.0)
            else:
                bond_cols[b] = COLOR_NONT

        legend = smi
        if matched_names:
            legend += " | " + ", ".join(matched_names[:4]) + ("..." if len(matched_names) > 4 else "")
        else:
            legend += " | (no alert match)"

        mols.append(mol)
        legends.append(legend)
        hl_atom_lists.append(atoms_all)
        hl_bond_lists.append(bonds_all)
        hl_atom_colors.append(atom_cols)
        hl_bond_colors.append(bond_cols)

    if not mols:
        raise ValueError("No valid molecules to draw after filtering.")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(320, 280),
        legends=legends,
        highlightAtomLists=hl_atom_lists,
        highlightBondLists=hl_bond_lists,
        highlightAtomColors=hl_atom_colors,
        highlightBondColors=hl_bond_colors,
        useSVG=False
    )
    img.save(out_png)
    print(f"Saved grid with Top-{len(mols)} highlighted fragments to {out_png} (sorted by '{score_col}')")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV with substructures and counts")
    ap.add_argument("--smiles_column", default="substructure_smiles", help="Column name containing SMILES")
    ap.add_argument("--alerts_csv", required=True, help="CSV with columns: Name,SMARTS,Class")
    ap.add_argument("--top_k", type=int, default=10, help="Number of top substructures to include")
    ap.add_argument("--sort_by", default=None, help="Column to sort by (default: total_pred or pred_pos+pred_neg)")
    ap.add_argument("--mols_per_row", type=int, default=5, help="Molecules per row in the grid")
    ap.add_argument("--out_png", default="fragments_highlighted_topk_grid.png", help="Output PNG filename")
    args = ap.parse_args()

    grid_highlight_topk(args.input_csv, args.smiles_column, args.alerts_csv, args.top_k, args.sort_by, args.mols_per_row, args.out_png)