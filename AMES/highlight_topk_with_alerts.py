
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
        name = str(row["Name"]).strip() if "Name" in row else smarts
        klass = str(row["Class"]).strip().lower()
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        alerts.append((name, patt, klass))
    return alerts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk_csv", required=True, help="CSV with a 'smi' column for the top-k substructures")
    ap.add_argument("--alerts_csv", required=True, help="CSV with columns: Name,SMARTS,Class")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    topk = pd.read_csv(args.topk_csv)
    if "smi" not in topk.columns:
        raise ValueError("Expected a 'smi' column in --topk_csv")

    alerts = load_alerts(args.alerts_csv)
    if not alerts:
        raise ValueError("No valid SMARTS patterns loaded from alerts CSV.")

    # Colors: toxic (red), nontoxic (blue)
    color_map = {"toxic": (0.9, 0.0, 0.0), "nontoxic": (0.0, 0.2, 0.9)}

    # Process first k entries
    for i, row in topk.head(args.k).iterrows():
        smi = str(row["smi"]).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Accumulate atom indices per class
        toxic_atoms = set()
        nontoxic_atoms = set()
        legend_bits = []

        for name, patt, klass in alerts:
            matches = mol.GetSubstructMatches(patt)
            if not matches:
                continue
            if klass not in color_map:
                continue
            # record atoms
            for match in matches:
                if klass == "toxic":
                    toxic_atoms.update(match)
                else:
                    nontoxic_atoms.update(match)
            # add legend entry
            legend_bits.append(f"{name} ({klass})")

        # Prepare per-atom colors
        hi_atoms = list(sorted(toxic_atoms.union(nontoxic_atoms)))
        atom_cols = {}
        for a in toxic_atoms:
            atom_cols[a] = color_map["toxic"]
        for a in nontoxic_atoms:
            # mix if overlaps
            if a in atom_cols:
                # simple average color if overlap
                r1,g1,b1 = atom_cols[a]
                r2,g2,b2 = color_map["nontoxic"]
                atom_cols[a] = ((r1+r2)/2.0, (g1+g2)/2.0, (b1+b2)/2.0)
            else:
                atom_cols[a] = color_map["nontoxic"]

        # Draw
        d2d = Draw.MolDraw2DCairo(600, 400)
        d2d.drawOptions().addAtomIndices = False
        d2d.drawOptions().addBondIndices = False
        d2d.drawOptions().useBWAtomPalette() # neutral atom colors for clarity
        Chem.rdDepictor.Compute2DCoords(mol)
        d2d.DrawMolecule(mol, highlightAtoms=hi_atoms, highlightAtomColors=atom_cols)
        # Legend
        legend = f"Top-k fragment #{i+1}\nSMILES: {smi}"
        if legend_bits:
            legend += "\nAlerts: " + "; ".join(legend_bits[:6]) + (" ..." if len(legend_bits) > 6 else "")
        d2d.FinishDrawing()

        out_png = os.path.join(args.outdir, f"topk_{i+1:02d}.png")
        with open(out_png, "wb") as f:
            f.write(d2d.GetDrawingText())

    # also write a small index file listing outputs
    with open(os.path.join(args.outdir, "index.txt"), "w") as f:
        for j in range(min(args.k, len(topk))):
            f.write(f"topk_{j+1:02d}.png\n")

if __name__ == "__main__":
    main()
