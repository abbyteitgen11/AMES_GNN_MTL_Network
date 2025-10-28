from datetime import datetime
import faulthandler
import os
import pdb
import re
import sys
import h5py
import random
import markdown
import csv
import argparse
import logging
from collections import Counter, defaultdict
import json
import pickle
import math
import yaml
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import colorsys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.explain import GNNExplainer, PGExplainer, Explainer
from torch_geometric.utils import to_networkx
import networkx as nx
from networkx.drawing import nx_agraph
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw, AllChem, rdmolops
from rdkit.DataStructs import TanimotoSimilarity

from callbacks import set_up_callbacks
from count_model_parameters import count_model_parameters
from device import device
from graph_dataset import GraphDataSet
from compute_metrics import *
from data import load_data
#from BuildNN_GNN_MTL import BuildNN_GNN_MTL
from BuildNN_GNN_MTL_GINEConv import BuildNN_GNN_MTL
from masked_loss_function import masked_loss_function
from set_seed import set_seed
from MTLDataset import MTLDataset
from TaskSpecificGNN import TaskSpecificGNN

# Set seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    return parser.parse_args()


# Load structural alerts
def load_alerts():
    """
    alerts = [
        ("Alkyl esters of phosphonic or sulphonic acids", "C[OX2]P(=O)(O)O or C[OX2]S(=O)(=O)O"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Aromatic N-oxides", "[n+](=O)[O-]"),
        ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Simple aldehyde", "[CX3H1](=O)[#6]"),
        ("N-methylol derivatives", "[NX3]CO"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("S- or N- mustards", "N(CCCl)CCCl or S(CCCl)CCCl"),
        ("Acyl halides", "[CX3](=O)[F,Cl,Br,I]"),
        ("Propiolactones and propiosultones", "O=C1OCC1 or O=S1OCC1"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
        ("Alkyl nitrite", "[CX4][OX2]N=O"),
        ("Quinones", "O=C1C=CC(=O)C=C1"),
        ("N-nitroso", "[NX3;H0,H1][NX2]=O"),
        ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
        ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
        ("Alpha, beta unsaturated carbonyls", "C=CC(=O)"),
        ("Isocyanate and isothiocyanate groups", "N=C=O or N=C=S"),
        ("Alkyl carbamate and thiocarbamate", "OC(=O)N or SC(=O)N"),
        ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
        ("Azide and triazene groups", "N=[N+]=[N-] or N=N-N"),
        ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
        ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12 or O=C1OC=CC2=CC=CC3=C21OCC3"),
        ("Halogenated benzene", "[c][F,Cl,Br,I]"),
        ("Halogenated polycyclic aromatic hydrocarbon", "[c1ccc2ccccc2c1][F,Cl,Br,I]"),
        ("Halogenated dibenzodioxins", "c1cc2Oc3c(cccc3Oc2cc1)[F,Cl,Br,I]"),
        ("Thiocarbonyls", "[CX3]=S"),
        ("Steroidal oestrogens", "C1CCC2C1(C)CCC3C2CCC4=CC(=O)CC=C34"),
        ("Trichloro/fluoro or tetrachloro/fluoro ethylene", "C([Cl,F])=C([Cl,F])[Cl,F]"),
        ("Pentachlorophenol", "c1(ccc(cc1Cl)Cl)Cl"),
        ("o-Phenylphenol", "c1ccccc1-c2ccccc2O"),
        ("Imidazole", "c1ncc[nH]1"),
        ("Dicarboximide", "O=C1NC(=O)C=C1"),
        ("Dimethylpyridine", "c1ccncc1C"),
        ("Michael acceptors", "C=CC=O"),
        ("Acrylamides", "C=CC(=O)N"),
        ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
        ("Polyhalogenated alkanes", "C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])"),
    ]
    """

    # Removing polycyclic hydrocarbons
    """
    alerts = [
        ("Alkyl esters of phosphonic or sulphonic acids", "C[OX2]P(=O)(O)O or C[OX2]S(=O)(=O)O"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Aromatic N-oxides", "[n+](=O)[O-]"),
        ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Simple aldehyde", "[CX3H1](=O)[#6]"),
        ("N-methylol derivatives", "[NX3]CO"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("S- or N- mustards", "N(CCCl)CCCl or S(CCCl)CCCl"),
        ("Acyl halides", "[CX3](=O)[F,Cl,Br,I]"),
        ("Propiolactones and propiosultones", "O=C1OCC1 or O=S1OCC1"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
        ("Alkyl nitrite", "[CX4][OX2]N=O"),
        ("Quinones", "O=C1C=CC(=O)C=C1"),
        ("N-nitroso", "[NX3;H0,H1][NX2]=O"),
        ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
        ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
        ("Alpha, beta unsaturated carbonyls", "C=CC(=O)"),
        ("Isocyanate and isothiocyanate groups", "N=C=O or N=C=S"),
        ("Alkyl carbamate and thiocarbamate", "OC(=O)N or SC(=O)N"),
        ("Azide and triazene groups", "N=[N+]=[N-] or N=N-N"),
        ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
        ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12 or O=C1OC=CC2=CC=CC3=C21OCC3"),
        ("Halogenated benzene", "[c][F,Cl,Br,I]"),
        ("Halogenated polycyclic aromatic hydrocarbon", "[c1ccc2ccccc2c1][F,Cl,Br,I]"),
        ("Halogenated dibenzodioxins", "c1cc2Oc3c(cccc3Oc2cc1)[F,Cl,Br,I]"),
        ("Thiocarbonyls", "[CX3]=S"),
        ("Steroidal oestrogens", "C1CCC2C1(C)CCC3C2CCC4=CC(=O)CC=C34"),
        ("Trichloro/fluoro or tetrachloro/fluoro ethylene", "C([Cl,F])=C([Cl,F])[Cl,F]"),
        ("Pentachlorophenol", "c1(ccc(cc1Cl)Cl)Cl"),
        ("o-Phenylphenol", "c1ccccc1-c2ccccc2O"),
        ("Imidazole", "c1ncc[nH]1"),
        ("Dicarboximide", "O=C1NC(=O)C=C1"),
        ("Dimethylpyridine", "c1ccncc1C"),
        ("Michael acceptors", "C=CC=O"),
        ("Acrylamides", "C=CC(=O)N"),
        ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
        ("Polyhalogenated alkanes", "C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])"),
    ]
    """

    # Excluding polycyclic hydrocarbons, nongenotoxic
    alerts = [
        ("Alkyl esters of phosphonic or sulphonic acids", "C[OX2]P(=O)(O)O or C[OX2]S(=O)(=O)O"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Aromatic N-oxides", "[n+](=O)[O-]"),
        ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Simple aldehyde", "[CX3H1](=O)[#6]"),
        ("N-methylol derivatives", "[NX3]CO"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("S- or N- mustards", "N(CCCl)CCCl or S(CCCl)CCCl"),
        ("Acyl halides", "[CX3](=O)[F,Cl,Br,I]"),
        ("Propiolactones and propiosultones", "O=C1OCC1 or O=S1OCC1"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
        ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
        ("Alkyl nitrite", "[CX4][OX2]N=O"),
        ("Quinones", "O=C1C=CC(=O)C=C1"),
        ("N-nitroso", "[NX3;H0,H1][NX2]=O"),
        ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
        ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
        ("Alpha, beta unsaturated carbonyls", "C=CC(=O)"),
        ("Isocyanate and isothiocyanate groups", "N=C=O or N=C=S"),
        ("Alkyl carbamate and thiocarbamate", "OC(=O)N or SC(=O)N"),
        ("Azide and triazene groups", "N=[N+]=[N-] or N=N-N"),
        ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
        ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12 or O=C1OC=CC2=CC=CC3=C21OCC3"),
        ("Michael acceptors", "C=CC=O"),
        ("Acrylamides", "C=CC(=O)N"),
        ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
    ]

    """
    # Simpler subset
    alerts = [
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Quinones", "O=C1C=CC(=O)C=C1"),
        ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
        ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
        ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
    ]
    """

    compiled = []

    for name, smarts in alerts:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            compiled.append((name, patt))
    return compiled


def compute_alert_confusion(global_smiles, alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks):
    n_mols = len(global_smiles)

    counts = Counter()

    for mol_id in range(n_mols):
        # Skip invalid labels (already filtered in your pipeline)
        present_alerts, _, _ = alerts_present_by_mol[mol_id]
        present_alerts = set(present_alerts)

        detected_alerts = set()
        for task in range(n_tasks):
            df_task = per_task_dfs[task][task]
            mol_df = df_task[df_task["mol_id"] == mol_id]
            detected_alerts.update(mol_df.loc[mol_df["alert_present"], "alert"].tolist())

        all_alerts = set([name for name, _ in alerts_compiled])

        for alert in all_alerts:
            present = alert in present_alerts
            detected = alert in detected_alerts

            if present and detected:
                counts["TP"] += 1
            elif not present and detected:
                counts["FP"] += 1
            elif present and not detected:
                counts["FN"] += 1
            else:
                counts["TN"] += 1

    return counts


def compute_per_alert_confusions(global_smiles, alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks):
    """Return dict: alert -> dict(TP, FN, FP, TN) aggregated across all strains."""
    results = defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0, "TN": 0})

    for mol_id, (present_alerts, _, _) in enumerate(alerts_present_by_mol):
        for alert, patt in alerts_compiled:
            # Ground truth
            gt = alert in present_alerts

            # Did model detect this alert in ANY strain?
            detected = False
            for task in range(n_tasks):
                df_task = per_task_dfs[task][task]
                mol_df = df_task[df_task['mol_id'] == mol_id]
                if not mol_df.empty and any((mol_df['alert'] == alert) & mol_df['alert_present']):
                    detected = True
                    break

            if gt and detected:
                results[alert]["TP"] += 1
            elif gt and not detected:
                results[alert]["FN"] += 1
            elif not gt and detected:
                results[alert]["FP"] += 1
            else:
                results[alert]["TN"] += 1

    return results


def plot_alert_confusion(counts):
    cm = [[counts["TN"], counts["FP"]],
          [counts["FN"], counts["TP"]]]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Detected", "Detected"],
                yticklabels=["No Alert", "Alert Present"])
    plt.xlabel("Model Detected")
    plt.ylabel("Ground Truth")
    plt.title("Structural Alert Detection Confusion Matrix")
    plt.tight_layout()
    plt.show()


# Compute overlap between substructure matches and important atoms
def compute_overlap_score(mol, smarts, highlighted_atoms):
    matches = mol.GetSubstructMatches(smarts)  # match_smarts(mol, smarts)
    if not matches:
        return 0.0, []

    highlighted_atoms = set(highlighted_atoms)
    scores = []

    for match in matches:
        match_set = set(match)
        overlap = len(match_set & highlighted_atoms) / len(match_set)
        scores.append(overlap)

    return max(scores), matches


def evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, correct_val, correct_val_overall):
    rows = []

    for i, (smiles, imp_dict, pred, label, label_overall) in enumerate(
            zip(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall)):
        # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mol = Chem.MolFromSmiles(smiles)
        for name, smarts in alerts:
            tight_score, _ = compute_overlap_score(mol, smarts, imp_dict["tight"])
            loose_score, _ = compute_overlap_score(mol, smarts, imp_dict["loose"])

            rows.append({
                "mol_id": i,
                "alert": name,
                "tight_score": tight_score,
                "loose_score": loose_score,
                "prediction": pred,
                "label": label,
                "label_overall": label_overall,
            })

    return pd.DataFrame(rows)


# Plotting
def draw_with_colors(mol, highlight_atoms, highlight_atom_colors,
                     highlight_bonds, highlight_bond_colors, size=(300, 300)):
    # ensure 2D coords exist
    if not mol.GetNumConformers():
        AllChem.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.useBWAtomPalette = False
    opts.highlightBondWidthMultiplier = 8  # make highlighted bonds thicker

    atom_cols = {int(k): tuple(v) for k, v in highlight_atom_colors.items()}
    bond_cols = {int(k): tuple(v) for k, v in highlight_bond_colors.items()}

    ha = sorted({int(i) for i in highlight_atoms})
    hb = sorted({int(i) for i in highlight_bonds})

    ha = [int(i) for i in highlight_atoms if 0 <= int(i) < mol.GetNumAtoms()]
    hb = [int(i) for i in highlight_bonds if 0 <= int(i) < mol.GetNumBonds()]
    atom_cols = {int(k): tuple(float(x) for x in v) for k, v in highlight_atom_colors.items() if
                 0 <= int(k) < mol.GetNumAtoms()}
    bond_cols = {int(k): tuple(float(x) for x in v) for k, v in highlight_bond_colors.items() if
                 0 <= int(k) < mol.GetNumBonds()}

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=ha,
        highlightBonds=hb,
        highlightAtomColors=atom_cols,
        highlightBondColors=bond_cols
    )
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png))


# Plot in grid
def plot_group(imgs, title, alert_color_dict, present_alerts, ncols=5):
    if not imgs:
        print(f"No molecules in {title}")
        return
    nrows = -(-len(imgs) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for ax, (img, mol_id, pred, label) in zip(axes, imgs):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Mol {mol_id}\nPred: {pred}, Label: {label}", fontsize=8)

    for ax in axes[len(imgs):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=16)

    # Legend
    legend_elements = [Patch(facecolor=alert_color_dict[name], edgecolor='k', label=name) for name in present_alerts]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()


def blank_image(size=(300, 300), color=(255, 255, 255)):
    return Image.new('RGB', size, color)


def hstack_images(imgs, pad=6, bg=(255, 255, 255)):
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths) + pad * (len(imgs) - 1)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h), bg)
    x = 0
    for im in imgs:
        new_im.paste(im, (x, (max_h - im.size[1]) // 2))
        x += im.size[0] + pad
    return new_im


import matplotlib.cm as cm
import numpy as np


def generate_alert_colors(alerts):
    reserved = [
        (1.0, 0.3, 0.3),  # red
        (1.0, 0.8, 0.3),  # orange
        (0.8, 0.8, 0.8),  # gray
    ]
    reserved = np.array(reserved)

    n = len(alerts)
    # use HSV evenly spaced colors
    hsv = [(i / n, 0.75, 0.95) for i in range(n)]
    rgb_candidates = [tuple(colorsys.hsv_to_rgb(*h)) for h in hsv]

    safe_colors = []
    for cand in rgb_candidates:
        cand_arr = np.array(cand)
        dists = np.linalg.norm(reserved - cand_arr, axis=1)
        if np.all(dists > 0.25):  # threshold to avoid looking too similar
            safe_colors.append(cand)
        else:
            # tweak hue slightly if too close
            new = ((cand[0] * 0.7 + 0.3), (cand[1] * 0.7), (cand[2] * 0.7))
            safe_colors.append(new)

    return {name: safe_colors[i] for i, (name, _) in enumerate(alerts)}


# Plot summary
def assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles,
                              alerts_compiled, output_dir, alerts_list):
    # cmap = cm.get_cmap("tab20", len(alerts_compiled))
    # alert_colors = {name: tuple(cmap(i)[:3]) for i, (name, _) in enumerate(alerts_compiled)}

    alert_colors = generate_alert_colors(alerts_compiled)

    os.makedirs(os.path.join(output_dir, 'summary_rows'), exist_ok=True)

    n_tasks = len(per_task_dfs)
    n_mols = len(global_smiles)
    cell_size = (300, 300)

    alerts_present_by_mol = []
    for s in global_smiles:
        # mol = Chem.AddHs(Chem.MolFromSmiles(s))
        mol = Chem.MolFromSmiles(s)
        atom_highlights = {}
        bond_highlights = {}
        present = []
        for name, patt in alerts_compiled:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                present.append(name)
                for match in matches:
                    for a in match:
                        atom_highlights[a] = (0.8, 0.8, 0.8)
                    for bond in mol.GetBonds():
                        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        if a1 in match and a2 in match:
                            bond_highlights[bond.GetIdx()] = (0.6, 0.6, 0.6)
        alerts_present_by_mol.append((present, atom_highlights, bond_highlights))

    all_row_imgs = []
    rows_correct_toxic = []
    rows_correct_nontoxic = []
    rows_incorrect = []

    for mol_id in range(n_mols):
        df0 = per_task_dfs[0]
        df_0 = df0[0]
        overall_rows = df_0[df_0['mol_id'] == mol_id]
        if overall_rows.empty:
            continue
        overall_label = int(overall_rows.iloc[0]['label_overall'])
        if overall_label == -1:
            continue

        mol = Chem.MolFromSmiles(global_smiles[mol_id])
        # mol = Chem.AddHs(Chem.MolFromSmiles(global_smiles[mol_id]))

        strain_cells = []
        for task in range(n_tasks):
            pdf = per_task_dfs[task]
            pdf_task = pdf[task]
            mol_df = pdf_task[pdf_task['mol_id'] == mol_id]
            if mol_df.empty:
                strain_cells.append(blank_image(cell_size))
                continue
            per_task_labels_t = per_task_labels[task]
            correct_label = int(per_task_labels_t[task][mol_id])
            per_task_preds_t = per_task_preds[task]
            pred = int(per_task_preds_t[task][mol_id])
            if correct_label == -1:
                im = blank_image(cell_size)
            else:
                highlight_atoms, atom_colors, highlight_bonds, bond_colors = [], {}, [], {}
                for _, row in mol_df.iterrows():
                    if row['alert_present']:
                        name = row['alert']
                        patt = next((p for n, p in alerts_compiled if n == name), None)
                        if patt is None:
                            continue
                        matches = mol.GetSubstructMatches(patt)
                        color = alert_colors.get(name, (0.5, 0.5, 0.5))
                        for match in matches:
                            for a in match:
                                highlight_atoms.append(a)
                                atom_colors[a] = color
                            for bond in mol.GetBonds():
                                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                                if a1 in match and a2 in match:
                                    bid = bond.GetIdx()
                                    highlight_bonds.append(bid)
                                    bond_colors[bid] = color

                # Add important atoms for this strain
                imp = per_task_impatoms[task]
                imp_t = imp[task]
                tight = imp_t[mol_id]['tight'] if mol_id < len(imp_t) else []
                loose = imp_t[mol_id]['loose'] if mol_id < len(imp_t) else []
                for a in tight:
                    atom_colors[a] = (1.0, 0.3, 0.3)  # red
                    highlight_atoms.append(a)
                for a in loose:
                    if a not in atom_colors:
                        atom_colors[a] = (1.0, 0.8, 0.3)  # orange
                    highlight_atoms.append(a)

                im = draw_with_colors(mol, highlight_atoms, atom_colors, highlight_bonds, bond_colors, size=cell_size)
                draw = ImageDraw.Draw(im)
                try:
                    font = ImageFont.truetype('DejaVuSans.ttf', 14)
                except Exception:
                    font = ImageFont.load_default()
                text = f"P:{pred} / L:{correct_label}"
                draw.rectangle([(0, 0), (im.size[0], 18)], fill=(255, 255, 255))
                draw.text((4, 0), text, fill=(0, 0, 0), font=font)
            strain_cells.append(im)

        preds = []
        for t in range(n_tasks):
            preds_t = per_task_preds[t]
            preds.append(int(preds_t[t][mol_id]))
        consensus = 1 if any(preds) else 0
        cons_im = blank_image(cell_size)
        d = ImageDraw.Draw(cons_im)
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 18)
        except Exception:
            font = ImageFont.load_default()
        d.text((10, 10), f"Consensus: {consensus}", fill=(0, 0, 0), font=font)
        d.text((10, 40), f"Overall label: {overall_label}", fill=(0, 0, 0), font=font)

        present, atom_highlights, bond_highlights = alerts_present_by_mol[mol_id]
        im_alerts_present = draw_with_colors(mol, list(atom_highlights.keys()), atom_highlights,
                                             list(bond_highlights.keys()), bond_highlights, size=cell_size)

        # union_atoms, atom_cols = set(), {}
        # for t in range(n_tasks):
        #    imp = per_task_impatoms[t]
        #    imp_t = imp[t]
        #    tight = imp_t[mol_id]['tight'] if mol_id < len(imp_t) else []
        #    loose = imp_t[mol_id]['loose'] if mol_id < len(imp_t) else []
        #    for a in tight:
        #        union_atoms.add(a)
        #        atom_cols[a] = (1.0, 0.3, 0.3)
        #    for a in loose:
        #        if a not in atom_cols:
        #            atom_cols[a] = (1.0, 0.8, 0.3)
        #        union_atoms.add(a)
        # im_imp = draw_with_colors(mol, list(union_atoms), atom_cols, [], {}, size=cell_size)

        row_imgs = strain_cells + [cons_im, im_alerts_present]
        row_concat = hstack_images(row_imgs, pad=4)

        # outfn = os.path.join(output_dir, 'summary_rows', f'mol_{mol_id:04d}_cons_{consensus}_ovl_{overall_label}.png')
        # row_concat.save(outfn)
        # all_row_imgs.append(row_concat.convert('RGB'))
        # print(f"Saved summary row for mol {mol_id} -> {outfn}")

        if consensus == 1 and overall_label == 1:
            rows_correct_toxic.append(row_concat)
        elif consensus == 0 and overall_label == 0:
            rows_correct_nontoxic.append(row_concat)
        else:
            rows_incorrect.append(row_concat)

    counts = compute_alert_confusion(global_smiles, alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks)
    plot_alert_confusion(counts)
    print(counts)

    per_alert_confusions = compute_per_alert_confusions(global_smiles, alerts_compiled, alerts_present_by_mol,
                                                        per_task_dfs, n_tasks)

    # Convert to DataFrame
    records = []
    records_summary = []
    for alert, c in per_alert_confusions.items():
        tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        records_summary.append({"alert": alert, "true positive": tp, "false negative": fn})
        records.append({"alert": alert, "sensitivity": sensitivity, "precision": precision})

    df_alert_perf = pd.DataFrame(records).sort_values("sensitivity", ascending=False)

    df_alert_summary = pd.DataFrame(records_summary).sort_values("true positive", ascending=False)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = np.arange(len(df_alert_perf))

    plt.bar(x - bar_width / 2, df_alert_perf["sensitivity"], width=bar_width, label="Sensitivity (Recall)")
    # plt.bar(x + bar_width / 2, df_alert_perf["precision"], width=bar_width, label="Precision")

    plt.xticks(x, df_alert_perf["alert"], rotation=90)
    plt.ylabel("Score")
    plt.title("Per-Alert Detection Performance (Global, all strains)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    x = np.arange(len(df_alert_summary))

    plt.bar(x - bar_width / 2, df_alert_summary["true positive"], width=bar_width, label="True positive")
    plt.bar(x + bar_width / 2, df_alert_summary["false negative"], width=bar_width, label="False negative")

    plt.xticks(x, df_alert_perf["alert"], rotation=90)
    plt.ylabel("Count")
    plt.title("Per-Alert Detection Performance (Global, all strains)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Strain-specific structural alert detection analysis ===
    detection_freqs = compute_alert_detection_by_strain(
        global_smiles,
        alerts_compiled,
        alerts_present_by_mol,
        per_task_dfs,
        n_tasks
    )

    #plot_alert_detection_heatmap(detection_freqs_ordered, order)

    # Save numerical table
    #csv_path = os.path.join(args.output_dir, "strain_specific_alert_detection_frequencies.csv")
    #detection_freqs.to_csv(csv_path)
    #print(f"✅ Saved strain-specific alert detection frequencies → {csv_path}")


    # === Overall alert detection performance ===
    df_perf = compute_overall_alert_performance(
        alerts_compiled,
        alerts_present_by_mol,
        per_task_dfs,
        n_tasks
    )

    output_dir = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output'
    df_perf = compute_overall_alert_performance(alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks)
    detection_freqs = compute_detection_frequencies(alerts_compiled, per_task_dfs, n_tasks)
    order, _ = plot_alert_performance_bars(df_perf, detection_freqs, output_dir)
    #plot_alert_detection_heatmap(detection_freqs, order, output_dir)


    df_summary = summarize_alert_strain_stats_to_csv(
        alerts_compiled=alerts_compiled,
        alerts_present_by_mol=alerts_present_by_mol,
        per_task_dfs=per_task_dfs,
        n_tasks=n_tasks,
        output_dir=output_dir
    )

    analyze_per_atom_overlap_by_alert(
        per_task_impatoms, alerts_compiled, global_smiles, output_dir
    )

    # Compute mean toxic overlap per alert per strain
    overlap_scores = compute_toxic_overlap_by_strain(
        alerts_compiled, per_task_dfs, n_tasks, alerts_present_by_mol
    )

    # Plot it using the same alert ordering as the bar plot
    plot_toxic_overlap_heatmap(overlap_scores, order, output_dir=output_dir)

    # Save to CSV
    SA_alert_summary_csv = os.path.join(output_dir, f"SA_summary.csv")
    df_alert_summary.to_csv(SA_alert_summary_csv, index=False)

    # Save category PDFs
    outdir = os.path.join(output_dir, "summary_rows")
    os.makedirs(outdir, exist_ok=True)

    save_rows_to_pdf(rows_correct_toxic,
                     os.path.join(outdir, "summary_correct_toxic.pdf"),
                     alert_colors)
    save_rows_to_pdf(rows_correct_nontoxic,
                     os.path.join(outdir, "summary_correct_nontoxic.pdf"),
                     alert_colors)
    save_rows_to_pdf(rows_incorrect,
                     os.path.join(outdir, "summary_incorrect.pdf"),
                     alert_colors)


def save_rows_to_pdf(row_imgs, pdf_path, alert_colors, rows_per_page=20):
    if not row_imgs:
        return
    row_imgs = [im.convert("RGB") for im in row_imgs]
    page_imgs = []
    for i in range(0, len(row_imgs), rows_per_page):
        batch = row_imgs[i:i + rows_per_page]
        widths, heights = zip(*(im.size for im in batch))
        page_w = max(widths)
        page_h = sum(heights)
        page = Image.new("RGB", (page_w, page_h), (255, 255, 255))
        y = 0
        for im in batch:
            page.paste(im, (0, y))
            y += im.size[1]
        page_imgs.append(page)

    # Add legend as last page
    legend_page = make_legend_page(alert_colors)
    page_imgs.append(legend_page)

    page_imgs[0].save(pdf_path, save_all=True, append_images=page_imgs[1:])
    print(f"Saved PDF: {pdf_path} ({len(page_imgs)} pages)")


def make_legend_page(alert_colors, size=(1200, 1600)):
    page = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    x, y = 50, 50
    # structural alerts
    draw.text((x, y), "Structural Alerts:", fill=(0, 0, 0), font=font)
    y += 40
    for name, color in alert_colors.items():
        rgb = tuple(int(c * 255) for c in color)
        draw.rectangle([x, y, x + 40, y + 40], fill=rgb)
        draw.text((x + 50, y), name, fill=(0, 0, 0), font=font)
        y += 50
        if y > size[1] - 50:
            y = 90
            x += 400

    return page

def compute_alert_detection_by_strain(global_smiles, alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks):
    """
    Returns a DataFrame of shape [n_alerts x n_strains] with values =
    fraction of molecules in which each alert was detected by each strain.
    """
    all_alerts = [name for name, _ in alerts_compiled]
    strain_names = [f"strain_{i+1}" for i in range(n_tasks)]

    # Initialize counts
    detection_counts = pd.DataFrame(0, index=all_alerts, columns=strain_names, dtype=float)
    n_mols = len(global_smiles)

    for mol_id, (present_alerts, _, _) in enumerate(alerts_present_by_mol):
        for task in range(n_tasks):
            df_task = per_task_dfs[task][task]
            mol_df = df_task[df_task["mol_id"] == mol_id]

            # Alerts detected by this strain in this molecule
            detected_alerts = mol_df.loc[mol_df["alert_present"], "alert"].unique().tolist()
            for alert in detected_alerts:
                if alert in all_alerts:
                    detection_counts.loc[alert, f"strain_{task+1}"] += 1

    # Normalize by total number of molecules → fraction
    detection_freqs = detection_counts / float(n_mols)
    return detection_freqs


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict


def compute_overall_alert_performance(alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks):
    """
    Compute overall alert-level metrics:
      - % correctly identified = (alert present & overlap > 0) / (alert present)
      - mean overlap score (overall)
      - mean overlap score for toxic and non-toxic molecules
    Returns DataFrame indexed by alert name.
    """
    all_alerts = [name for name, _ in alerts_compiled]
    stats = {
        a: {
            "n_present": 0,
            "n_detected": 0,
            "overlaps": [],
            "overlaps_toxic": [],
            "overlaps_nontoxic": [],
        }
        for a in all_alerts
    }

    for mol_id, (present_alerts, _, label_list) in enumerate(alerts_present_by_mol):
        is_toxic = any(l == 1 for l in label_list)
        mol_overlaps = defaultdict(list)
        mol_detected = defaultdict(bool)

        # Loop through all tasks/strains
        for task in range(n_tasks):
            df_task = per_task_dfs[task][task]
            mol_df = df_task[df_task["mol_id"] == mol_id]

            for _, row in mol_df.iterrows():
                a = row["alert"]
                if a not in stats or not row["alert_present"]:
                    continue
                overlap = row.get("tight_score", 0.0)
                mol_overlaps[a].append(overlap)
                if overlap > 0:
                    mol_detected[a] = True

        # Aggregate per alert
        for alert in present_alerts:
            if alert not in stats:
                continue
            stats[alert]["n_present"] += 1
            if mol_detected[alert]:
                stats[alert]["n_detected"] += 1
            if alert in mol_overlaps:
                stats[alert]["overlaps"].extend(mol_overlaps[alert])
                if is_toxic:
                    stats[alert]["overlaps_toxic"].extend(mol_overlaps[alert])
                else:
                    stats[alert]["overlaps_nontoxic"].extend(mol_overlaps[alert])

    # Convert to DataFrame
    rows = []
    for alert, v in stats.items():
        n_pres = v["n_present"]
        n_det = v["n_detected"]
        pct_correct = (n_det / n_pres * 100) if n_pres > 0 else 0
        mean_overlap = np.mean(v["overlaps"]) if v["overlaps"] else 0
        mean_overlap_tox = np.mean(v["overlaps_toxic"]) if v["overlaps_toxic"] else 0
        mean_overlap_non = np.mean(v["overlaps_nontoxic"]) if v["overlaps_nontoxic"] else 0
        rows.append({
            "alert": alert,
            "percent_correct": pct_correct,
            "mean_overlap": mean_overlap,
            "mean_overlap_toxic": mean_overlap_tox,
            "mean_overlap_nontoxic": mean_overlap_non
        })

    df_perf = pd.DataFrame(rows).set_index("alert")
    return df_perf


def compute_detection_frequencies(alerts_compiled, per_task_dfs, n_tasks):
    """
    Compute detection frequency for each alert per strain:
    fraction of all molecules (not just ones where alert_present=True)
    where alert overlap > 0.
    """
    all_alerts = [name for name, _ in alerts_compiled]
    detection_freqs = pd.DataFrame(0.0, index=all_alerts, columns=[f"Strain {i+1}" for i in range(n_tasks)])

    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()
        df_task["alert_detected"] = df_task["tight_score"] > 0

        for alert in all_alerts:
            df_alert = df_task[df_task["alert"] == alert]
            if len(df_alert) == 0:
                continue

            # detection fraction across *all* molecules, not just where alert_present=True
            mol_detected = df_alert.groupby("mol_id")["alert_detected"].max()
            detection_freqs.loc[alert, f"Strain {task+1}"] = mol_detected.mean()  # average across all mols

    return detection_freqs

def compute_toxic_overlap_by_strain(alerts_compiled, per_task_dfs, n_tasks, alerts_present_by_mol):
    """
    Computes mean overlap score (tight_score) for toxic molecules only,
    per alert × strain.
    Returns a DataFrame with alerts as rows, strains as columns.
    """
    all_alerts = [name for name, _ in alerts_compiled]
    overlap_scores = pd.DataFrame(0.0, index=all_alerts, columns=[f"Strain {i+1}" for i in range(n_tasks)])

    # Identify toxic molecule IDs
    toxic_mols = [i for i, (_, _, labels) in enumerate(alerts_present_by_mol) if any(l == 1 for l in labels)]

    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()
        df_task = df_task[df_task["mol_id"].isin(toxic_mols)]

        for alert in all_alerts:
            df_alert = df_task[(df_task["alert"] == alert) & (df_task["alert_present"])]
            if len(df_alert) == 0:
                continue
            mean_overlap = df_alert["tight_score"].mean()
            overlap_scores.loc[alert, f"Strain {task+1}"] = mean_overlap

    return overlap_scores



def plot_alert_performance_bars(df_perf, detection_freqs=None, output_dir=None):
    """
    Plot horizontal bars for:
      - % correctly identified
      - mean overlap (toxic)
      - mean overlap (non-toxic)
    Alerts ordered by descending mean overlap (toxic).
    """
    order = df_perf.sort_values("mean_overlap_toxic", ascending=False).index
    df_perf = df_perf.loc[order]
    if detection_freqs is not None:
        detection_freqs = detection_freqs.loc[order]

    #fig, ax = plt.subplots(figsize=(12, 0.4 * len(df_perf)))  # longer and skinnier
    fig, ax = plt.subplots(figsize=(5, 15))

    y = np.arange(len(df_perf))
    barh_kwargs = dict(height=0.25, edgecolor="black")

    ax.barh(y - 0.25, df_perf["percent_correct"], color="#6baed6", label="% Correctly Identified", **barh_kwargs)
    ax.barh(y, df_perf["mean_overlap_toxic"] * 100, color="#fd8d3c", label="Mean Overlap (Toxic)", **barh_kwargs)
    ax.barh(y + 0.25, df_perf["mean_overlap_nontoxic"] * 100, color="#969696", label="Mean Overlap (Non-Toxic)", **barh_kwargs)

    ax.set_yticks(y)
    ax.set_yticklabels(df_perf.index)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage / Overlap ×100")
    ax.set_title("Structural Alert Detection Performance (Overlap > 0 Definition)")
    #ax.legend()
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, "alert_performance_bars.png")
        plt.savefig(outpath, dpi=300, transparent=True)
        print(f"✅ Saved alert performance bars → {outpath}")
        plt.close()
    else:
        plt.show()

    return order, detection_freqs

def plot_toxic_overlap_heatmap(overlap_scores, order, output_dir=None):
    """
    Plots a heatmap showing mean toxic overlap per alert per strain.
    Alerts are ordered to match the bar plot ordering (descending toxic overlap).
    """
    overlap_scores = overlap_scores.loc[order]

    plt.figure(figsize=(8,15))
    sns.heatmap(
        overlap_scores * 100,  # convert to percentage for readability
        cmap="YlOrRd",
        cbar_kws={"label": "Mean Overlap (Toxic, %)"},
        linewidths=0.5,
        linecolor="lightgray",
        annot=True,
        fmt=".1f"
    )

    plt.title("Mean Overlap Score (Toxic Molecules Only) per Strain")
    plt.xlabel("Strain")
    plt.ylabel("Structural Alert")
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, "toxic_overlap_by_strain_heatmap.png")
        plt.savefig(outpath, dpi=600, transparent=True)  # high-res + transparent bg
        print(f"✅ Saved toxic-overlap heatmap → {outpath}")
        plt.close()
    else:
        plt.show()


def plot_alert_detection_heatmap(detection_freqs, order, output_dir=None):
    """
    Heatmap showing fraction of molecules where each alert was detected (overlap > 0),
    grouped by strain, ordered by mean toxic overlap.
    """
    detection_freqs = detection_freqs.loc[order]

    fig_height = max(10, 0.4 * len(detection_freqs))
    #plt.figure(figsize=(12, fig_height))
    plt.figure(figsize=(8, 15))

    sns.heatmap(
        detection_freqs,
        cmap="viridis",
        cbar_kws={"label": "Fraction of Molecules (Overlap > 0)"},
        linewidths=0.5,
        linecolor="lightgray",
        annot=True,
        fmt=".2f"
    )

    plt.title("Structural Alerts Detected per Strain (Overlap > 0 Definition)")
    plt.xlabel("Strain")
    plt.ylabel("Structural Alert")
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, "strain_specific_alert_detection_heatmap.png")
        plt.savefig(outpath, dpi=600, transparent=True)
        print(f"✅ Saved strain-specific alert detection heatmap → {outpath}")
        plt.close()
    else:
        plt.show()

def summarize_alert_strain_stats_to_csv(
    alerts_compiled,
    alerts_present_by_mol,
    per_task_dfs,
    n_tasks,
    output_dir=None
):
    """
    For each alert × strain, compute:
      - number of molecules with the alert
      - number correctly identified (overlap > 0)
      - mean overlap (toxic only)
      - mean overlap (non-toxic only)
    and save to a CSV file.
    """

    all_alerts = [name for name, _ in alerts_compiled]
    rows = []

    # Identify toxic vs non-toxic molecules
    toxic_mols = {i for i, (_, _, labels) in enumerate(alerts_present_by_mol) if any(l == 1 for l in labels)}
    nontoxic_mols = {i for i, (_, _, labels) in enumerate(alerts_present_by_mol) if all(l != 1 for l in labels)}

    for task in range(n_tasks):
        strain_name = f"Strain {task+1}"
        df_task = per_task_dfs[task][task].copy()
        df_task["alert_detected"] = df_task["tight_score"] > 0

        for alert in all_alerts:
            df_alert = df_task[df_task["alert"] == alert]

            # Only consider molecules where the alert is actually present
            df_present = df_alert[df_alert["alert_present"] == True]
            if len(df_present) == 0:
                continue

            n_total = len(df_present["mol_id"].unique())
            n_detected = len(df_present[df_present["alert_detected"]]["mol_id"].unique())

            # Toxic-only overlap mean
            df_toxic = df_present[df_present["mol_id"].isin(toxic_mols)]
            mean_overlap_tox = df_toxic["tight_score"].mean() if len(df_toxic) > 0 else 0

            # Non-toxic-only overlap mean
            df_non = df_present[df_present["mol_id"].isin(nontoxic_mols)]
            mean_overlap_non = df_non["tight_score"].mean() if len(df_non) > 0 else 0

            rows.append({
                "alert": alert,
                "strain": strain_name,
                "n_molecules_with_alert": n_total,
                "n_correctly_identified": n_detected,
                "percent_correct": (n_detected / n_total * 100) if n_total > 0 else 0,
                "mean_overlap_toxic_%": mean_overlap_tox * 100,
                "mean_overlap_nontoxic_%": mean_overlap_non * 100
            })

    df_summary = pd.DataFrame(rows)

    if output_dir:
        outpath = os.path.join(output_dir, "alert_strain_summary.csv")
        df_summary.to_csv(outpath, index=False)
        print(f"✅ Saved alert × strain summary → {outpath}")
    else:
        print(df_summary.head())

    return df_summary

from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

import os, io
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import cm
from PIL import Image

def analyze_per_atom_overlap_by_alert(per_task_impatoms, alerts_compiled, global_smiles, output_dir):
    """
    Compute per-atom overlap frequencies between structural alerts and important atoms across tasks.

    - Counts each (task, molecule) instance where the alert occurs.
    - For each atom matched by the alert, counts how often it is included in the 'tight' or 'loose' important atom sets.
    - Produces PNG visualizations and a CSV summary (alert_atom_overlap_summary.csv).
    """
    import os
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    alert_instance_counts = defaultdict(int)
    alert_atom_counts_tight = defaultdict(lambda: defaultdict(int))
    alert_atom_counts_loose = defaultdict(lambda: defaultdict(int))

    # Iterate over tasks and molecules (task_dict is {task_id: [mol_dicts]})
    for task_dict in per_task_impatoms:
        task_id = list(task_dict.keys())[0]
        imp_list = task_dict[task_id]
        for mol_id, smi in enumerate(global_smiles):
            if mol_id >= len(imp_list):
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            imp_entry = imp_list[mol_id] if mol_id < len(imp_list) else {"tight": [], "loose": []}
            tight_set = set(imp_entry.get('tight', []))
            loose_set = set(imp_entry.get('loose', []))

            for alert_name, patt in alerts_compiled:
                if patt is None:
                    continue
                matches = mol.GetSubstructMatches(patt)
                if not matches:
                    continue
                alert_atom_indices = sorted(set([idx for match in matches for idx in match]))
                if not alert_atom_indices:
                    continue

                alert_instance_counts[alert_name] += 1
                for idx in alert_atom_indices:
                    if idx in tight_set:
                        alert_atom_counts_tight[alert_name][idx] += 1
                    if idx in loose_set:
                        alert_atom_counts_loose[alert_name][idx] += 1

    # Normalize frequencies and save plots + CSV
    plot_dir = os.path.join(output_dir, "alert_atom_overlap_plots")
    os.makedirs(plot_dir, exist_ok=True)

    records = []
    for alert_name, instance_count in alert_instance_counts.items():
        if instance_count == 0:
            continue
        tight_counts = alert_atom_counts_tight.get(alert_name, {})
        loose_counts = alert_atom_counts_loose.get(alert_name, {})
        # find a representative molecule for visualization
        rep_mol = None
        rep_idx = None
        for mol_id, smi in enumerate(global_smiles):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            patt = next((p for n, p in alerts_compiled if n == alert_name), None)
            if patt and mol.HasSubstructMatch(patt):
                rep_mol = mol
                rep_idx = mol_id
                break
        if rep_mol is None:
            continue

        atom_scores = np.zeros(rep_mol.GetNumAtoms(), dtype=float)
        # use max(tight, loose) for visualization intensity
        for idx in range(rep_mol.GetNumAtoms()):
            t = tight_counts.get(idx, 0) / float(instance_count)
            l = loose_counts.get(idx, 0) / float(instance_count)
            atom_scores[idx] = max(t, l)
            records.append({
                'alert': alert_name,
                'rep_mol_id': rep_idx,
                'atom_index': idx,
                'tight_overlap_freq': tight_counts.get(idx, 0) / float(instance_count),
                'loose_overlap_freq': loose_counts.get(idx, 0) / float(instance_count),
            })

        # normalize to [0,1]
        max_v = atom_scores.max()
        norm_vals = atom_scores / (max_v + 1e-12) if max_v > 0 else atom_scores
        atom_colors = {i: (1.0, 1.0 - float(norm_vals[i]), 1.0 - float(norm_vals[i])) for i in range(rep_mol.GetNumAtoms())}

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            rep_mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightAtomRadii={i: 0.4 for i in atom_colors.keys()},
        )
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        outpath = os.path.join(plot_dir, f"{alert_name.replace('/', '_')}_atom_overlap.png")
        with open(outpath, "wb") as fh:
            fh.write(png_bytes)

    # write CSV
    csv_path = os.path.join(output_dir, "alert_atom_overlap_summary.csv")
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

    print(f"✅ Per-atom overlap analysis complete. Plots saved to: {plot_dir}  CSV: {csv_path}")


def main():
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file = args.input_file

    with open(input_file, 'r') as input_stream:
        input_data = yaml.load(input_stream, Loader=yaml.Loader)

    # Set database path
    database_path = input_data.get("database", "./GraphDataBase_AMES")

    # The database is described with its own yaml file; so read it
    database_file = database_path + '/graph_description.yml'

    with open(database_file, 'r') as database_stream:
        database_data = yaml.load(database_stream, Loader=yaml.Loader)

    # Model parameters
    n_graph_convolution_layers = input_data.get("nGraphConvolutionLayers", 0)  # Number of graph convolutional layers
    n_node_neurons = input_data.get("nNodeNeurons", None)  # Number of neurons in GNN
    n_edge_neurons = input_data.get("nEdgeNeurons", None)  # Number of edges in GNN
    dropout_GNN = input_data.get("dropoutGNN", None)  # Dropout GNN
    momentum_batch_norm = input_data.get("momentumBatchNorm", None)  # Batch norm GNN

    n_shared_layers = input_data.get("nSharedLayers", 4)  # Number of layers in shared core
    n_target_specific_layers = input_data.get("nTargetSpecificLayers", 2)  # Number of layers in target specific core
    n_shared = input_data.get("nShared", None)  # Number of neurons in shared core
    n_target = input_data.get("nTarget", None)  # Number of neurons in target specific core
    dropout_shared = input_data.get("dropoutShared", None)  # Dropout in shared core
    dropout_target = input_data.get("dropoutTarget", None)  # Dropout in target specific core

    activation = input_data.get("ActivationFunction", "ReLU")  # Activation function
    weighted_loss_function = input_data.get("weightedCostFunction", False)
    w1 = input_data.get("w1", 1.0)
    w2 = input_data.get("w2", 1.0)
    w3 = input_data.get("w3", 1.0)
    w4 = input_data.get("w4", 1.0)
    w5 = input_data.get("w5", 1.0)
    if weighted_loss_function:
        # class_weights = {
        #    '98': {0: 1.0, 1: w1, -1: 0},
        #    '100': {0: 1.0, 1: w2, -1: 0},
        #    '102': {0: 1.0, 1: w3, -1: 0},
        #    '1535': {0: 1.0, 1: w4, -1: 0},
        #    '1537': {0: 1.0, 1: w5, -1: 0},
        # }
        class_weights = {
            '98': {0: 1.0, 1: 1.6599, -1: 0},
            '100': {0: 1.0, 1: 1.2982, -1: 0},
            '102': {0: 1.0, 1: 2.5973, -1: 0},
            '1535': {0: 1.0, 1: 4.8234, -1: 0},
            '1537': {0: 1.0, 1: 4.8740, -1: 0},
        }
    else:
        class_weights = {
            '98': {0: 1.0, 1: 1.0, -1: 0.0},
            '100': {0: 1.0, 1: 1.0, -1: 0.0},
            '102': {0: 1.0, 1: 1.0, -1: 0.0},
            '1535': {0: 1.0, 1: 1.0, -1: 0.0},
            '1537': {0: 1.0, 1: 1.0, -1: 0.0},
        }
    output_keys = ['98', '100', '102', '1535', '1537']

    # Graph information
    graph_type = database_data.get("graphType", "covalent")
    n_node_features = database_data.get("nNodeFeatures")
    edge_parameters = database_data.get("EdgeFeatures")
    bond_angle_features = database_data.get("BondAngleFeatures", True)
    dihedral_angle_features = database_data.get("DihedralFeatures", True)
    n_edge_features = 1  # 1 for distance features
    if bond_angle_features: n_edge_features += 1  # bond-angle feature
    if dihedral_angle_features: n_edge_features += 1  # dihedral-angle feature

    # Training parameters
    nEpochs = input_data.get("nEpochs", 10)  # Number of epochs
    nBatch = input_data.get("nBatch", 50)  # Batch size
    chkptFreq = input_data.get("nCheckpoint", 10)  # Checkpoint frequency
    seed = input_data.get("randomSeed", 42)  # Random seed
    nTrainMaxEntries = input_data.get("nTrainMaxEntries",
                                      None)  # Number of training examples to use (if not using whole dataset)
    nValMaxEntries = input_data.get("nValMaxEntries",
                                    None)  # Number of validation examples to use (if not using whole dataset)
    learningRate = input_data.get("learningRate", 0.0001)  # Learning rate
    weightedCostFunction = input_data.get("weightedCostFunction", None)  # Use weighted  cost function
    L2Regularization = input_data.get("L2Regularization", 0.005)  # L2 regularization coefficient
    loadModel = input_data.get("loadModel", False)
    loadOptimizer = input_data.get("loadOptimizer", False)
    useMolecularDescriptors = input_data.get("useMolecularDescriptors",
                                             False)  # Use molecular descriptors instead of graphs for comparison to original MTL paper

    trainDir = database_path + '/train/'
    valDir = database_path + '/validate/'
    testDir = database_path + '/test/'
    directories = [trainDir, valDir, testDir]

    n_inputs = 0

    # Read in graph data
    trainDataset = GraphDataSet(
        trainDir, nMaxEntries=nTrainMaxEntries, seed=seed
    )

    if nTrainMaxEntries:
        nTrain = nTrainMaxEntries
    else:
        nTrain = len(trainDataset)

    valDataset = GraphDataSet(
        valDir, nMaxEntries=nValMaxEntries, seed=seed
    )

    if nValMaxEntries:
        nValidation = nValMaxEntries
    else:
        nValidation = len(valDataset)

    testDataset = GraphDataSet(
        testDir, nMaxEntries=nValMaxEntries, seed=seed
    )

    g = torch.Generator()
    g.manual_seed(seed)

    # Build model
    model = BuildNN_GNN_MTL(n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features,
                            n_edge_features, dropout_GNN, momentum_batch_norm,
                            n_shared_layers, n_target_specific_layers, n_shared, n_target, dropout_shared,
                            dropout_target,
                            activation, useMolecularDescriptors, n_inputs)

    checkpoint = torch.load(
        '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output_seed_8_20_25/checkpoints/metrics_45_1.pt',
        map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model = model.to(device)

    per_task_dfs = []
    per_task_impatoms = []
    per_task_preds = []
    per_task_labels = []
    global_smiles = []

    for task_id in range(5):

        task = task_id
        model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers,
                      n_shared_layers, n_target_specific_layers, useMolecularDescriptors)

        task_model = TaskSpecificGNN(model, task_idx=task, model_args=model_args)
        task_model.eval()

        explainer = Explainer(
            model=task_model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='probs',
            ),
        )

        ####Loop through dataset
        node_masks_all = []
        smiles_list = []
        predictions = []
        important_atoms_per_mol = []
        correct_val = []
        correct_val_overall = []

        for i, data in enumerate(testDataset):  # limit if needed for speed
            data = data.to(device)
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                # global_feats=data.global_feats
            )

            with torch.no_grad():
                task_output = task_model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch,
                    # global_feats=data.global_feats
                )

                prediction = int(task_output.item() > 0.5)  # 1 = toxic, 0 = non-toxic
                predictions.append(prediction)

            edge_mask = explanation.edge_mask.detach().cpu().numpy()

            # Tight filter
            k_edges_tight = int(0.15 * edge_mask.size)  # max(8, int(0.15 * edge_mask.size))  # ~10–15%
            top_e_tight = np.argsort(-edge_mask)[:k_edges_tight]

            imp_edges_tight = data.edge_index[:, torch.tensor(top_e_tight, device=data.edge_index.device)]
            imp_nodes_tight = sorted(set(imp_edges_tight.view(-1).tolist()))

            G = to_networkx(data, to_undirected=True)
            sub_tight = G.subgraph(imp_nodes_tight).copy()
            # if sub_tight.number_of_nodes() > 0:
            #    lcc_tight = max(nx.connected_components(sub_tight), key=len)
            #    important_atoms_tight = sorted(list(lcc_tight))
            # else:
            #    important_atoms_tight = []
            # if sub_tight.number_of_nodes() > 0:
            #    important_atoms_tight = imp_nodes_tight
            # else:
            #    important_atoms_tight = []
            if sub_tight.number_of_nodes() > 0:
                # Keep *all* connected components, not just the largest
                comps = max(nx.connected_components(sub_tight), key=len)
                important_atoms_tight = sorted(list(comps))
                # comps = list(nx.connected_components(sub_tight))
                # important_atoms_tight = sorted(set().union(*comps))
            else:
                important_atoms_tight = []

            # Loose filter
            k_edges_loose = int(0.15 * edge_mask.size)  # max(8, int(0.15 * edge_mask.size))  # ~25–30%
            top_e_loose = np.argsort(-edge_mask)[:k_edges_loose]

            imp_edges_loose = data.edge_index[:, torch.tensor(top_e_loose, device=data.edge_index.device)]
            imp_nodes_loose = sorted(set(imp_edges_loose.view(-1).tolist()))

            sub_loose = G.subgraph(imp_nodes_loose).copy()
            if sub_loose.number_of_nodes() > 0:
                # Keep *all* connected components, not just the largest
                comps = max(nx.connected_components(sub_loose), key=len)
                important_atoms_loose = sorted(list(comps))
                # comps = list(nx.connected_components(sub_loose))
                # important_atoms_loose = sorted(set().union(*comps))
            else:
                important_atoms_loose = []
            # if sub_loose.number_of_nodes() > 0:
            #    important_atoms_loose = imp_nodes_loose
            # else:
            #    important_atoms_tight = []

            # Collect both sets
            important_atoms_per_mol.append({
                "tight": important_atoms_tight,
                "loose": important_atoms_loose
            })

            # Extract SMILES
            # CSV file with structure data
            csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv'
            df = pd.read_csv(csv_file)
            filepath = os.path.basename(data.file_name)

            molecule_index = molecule_index = int(
                re.search(r'(\d+)_', filepath).group(1))  # get molecule number from input file name
            smiles_column_index = 3
            correct_val_index = 1364 + task
            correct_val_overall_index = 1369

            # Extract the SMILES string from the specific row and column
            smiles_string = df.iloc[molecule_index - 1, smiles_column_index]
            smiles_list.append(smiles_string)

            correct = df.iloc[molecule_index - 1, correct_val_index]
            correct_val.append(correct)

            correct_overall = df.iloc[molecule_index - 1, correct_val_overall_index]
            correct_val_overall.append(correct_overall)

        per_task_impatoms.append({task_id: important_atoms_per_mol})
        per_task_preds.append({task_id: predictions})
        per_task_labels.append({task_id: correct_val})
        global_smiles = smiles_list

        alerts = load_alerts()

        df = evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, correct_val,
                             correct_val_overall)  # Compute overlap scores by comparing alerts and important nodes, store all in df
        df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
        per_task_dfs.append({task_id: df})

        summary = df.groupby("alert")[
            ["tight_score", "loose_score"]].mean().reset_index()  # compute average scores for each alert
        summary = summary.sort_values("loose_score", ascending=False)  # sort by loose score
        alerts_csv = os.path.join(args.output_dir, f"alerts_task_{task}.csv")  # write to csv
        summary.to_csv(alerts_csv, index=False)
        alerts_list = summary["alert"].tolist()  # <- canonical order for ALL plots
        x = np.arange(len(alerts_list))

        ##### Plot per-strain results
        plt.figure(figsize=(12, 6))
        plt.bar(x, summary["tight_score"], alpha=0.6)
        # plt.bar(x, summary["loose_score"], alpha=0.6, label="Loose")
        plt.xticks(x, alerts_list, rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title(f"Task {task}: GNN explanation overlap with functional alerts, all molecules")
        plt.legend()
        plt.tight_layout()
        plt.show()

        tight_means = (
            df.groupby(["alert", "prediction"])["tight_score"]
            .mean()
            .unstack(fill_value=0)  # ensures missing entries are filled
            .reindex(columns=[0, 1], fill_value=0)  # guarantees both columns exist
            .reindex(alerts_list)  # enforces alert order
        )

        mean_tight_nontoxic = df.loc[df["prediction"] == 0, "tight_score"].mean()
        mean_tight_toxic = df.loc[df["prediction"] == 1, "tight_score"].mean()

        plt.figure(figsize=(12, 6))
        plt.bar(x, tight_means[0], width=0.5, color='blue', alpha=0.6,
                label=f"Nontoxic (pred=0), overall mean={mean_tight_nontoxic:.2f}")
        plt.bar(x, tight_means[1], width=0.5, color='red', alpha=0.6,
                label=f"Toxic (pred=1), overall mean={mean_tight_toxic:.2f}")
        plt.xticks(x, alerts_list, rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title(f"Task {task}: Mean overlap score by predicted toxicity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        loose_means = (
            df.groupby(["alert", "prediction"])["loose_score"]
            .mean()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
            .reindex(alerts_list)
        )

        mean_loose_nontoxic = df.loc[df["prediction"] == 0, "loose_score"].mean()
        mean_loose_toxic = df.loc[df["prediction"] == 1, "loose_score"].mean()

        plt.figure(figsize=(12, 6))
        plt.bar(x, loose_means[0], width=0.5, color='blue', alpha=0.6,
                label=f"Nontoxic (pred=0), overall mean={mean_loose_nontoxic:.2f}")
        plt.bar(x, loose_means[1], width=0.5, color='red', alpha=0.6,
                label=f"Toxic (pred=1), overall mean={mean_loose_toxic:.2f}")
        plt.xticks(x, alerts_list, rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title(f"Task {task}: Mean overlap score by predicted toxicity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        pivot_df = df.pivot(index="mol_id", columns="alert", values="loose_score")
        pivot_df = pivot_df.reindex(columns=alerts_list)

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, cmap="viridis", cbar_kws={"label": "Overlap score"})
        plt.title("Per-molecule overlap with functional alerts")
        plt.xlabel("Alert")
        plt.ylabel("Molecule")
        plt.tight_layout()
        plt.show()

        # Consider an alert present if tight_score > 0 or loose_score > 0
        df['alert_present'] = (df['tight_score'] > 0) | (df['loose_score'] > 0)
        # df['alert_present'] = (df['tight_score'] > 0)

        alert_summary = df.groupby("alert").agg(
            n_positive_predictions=("prediction", lambda x: ((x == 1) & df.loc[x.index, 'alert_present']).sum()),
            n_negative_predictions=("prediction", lambda x: ((x == 0) & df.loc[x.index, 'alert_present']).sum()),
            n_positive_labels=("label", lambda x: ((x == 1) & df.loc[x.index, 'alert_present']).sum()),
            n_negative_labels=("label", lambda x: ((x == 0) & df.loc[x.index, 'alert_present']).sum()),
        ).reset_index()

        alert_summary = alert_summary.set_index("alert").reindex(alerts_list).reset_index()

        # Save to CSV
        alert_summary_csv = os.path.join(args.output_dir, f"alert_summary_task_{task}.csv")
        alert_summary.to_csv(alert_summary_csv, index=False)

        # Plot
        plt.figure(figsize=(12, 6))
        alert_summary.plot(
            x="alert",
            y=["n_positive_predictions", "n_negative_predictions", "n_positive_labels", "n_negative_labels"],
            kind="bar",
            stacked=False,
            figsize=(12, 6)
        )
        plt.xticks(rotation=90)
        plt.ylabel("Count of molecules with alert")
        plt.title(f"Structural alerts: Predictions vs Labels (Task {task})")
        plt.tight_layout()
        plt.show()

        # Only include molecules with valid label_overall and label
        # valid_mols = df[(df['label_overall'] != -1) & (df['label'] != -1)]['mol_id'].unique()
        valid_mols = df['mol_id'].unique()
        # Use tab20 for distinct alert colors
        alert_names = df['alert'].unique()
        n_alerts = len(alert_names)
        cmap = cm.get_cmap('tab20', n_alerts)
        alert_color_dict = {alert_names[i]: cmap(i)[:3] for i in range(n_alerts)}
        alert_color_dict['Heterocyclic/polycyclic aromatic hydrocarbons'] = (0.8, 0.0, 0.8)
        present_alerts = set()

        correct_toxic_imgs = []
        correct_nontoxic_imgs = []
        incorrect_imgs = []

        for mol_id in valid_mols:
            # mol_df = df[(df['mol_id'] == mol_id) & (df['alert_present'])]
            mol_df = df[(df['mol_id'] == mol_id)]
            mol = Chem.MolFromSmiles(smiles_list[mol_id])
            # mol = Chem.AddHs(Chem.MolFromSmiles(smiles_list[mol_id]))
            # if mol is None or mol_df.empty:
            #    #ax.axis('off')
            #    continue

            highlight_atoms = []
            highlight_colors = {}
            highlight_bonds = []
            highlight_bond_colors = {}

            for _, row in mol_df.iterrows():
                if row['alert_present']:
                    name = row['alert']
                    smarts_pattern = next((p for n, p in alerts if n == name), None)
                    if smarts_pattern is None:
                        continue

                    present_alerts.add(name)

                    # Get RGB triple
                    r, g, b = alert_color_dict[name]
                    color = (float(r), float(g), float(b))

                    matches = mol.GetSubstructMatches(smarts_pattern)
                    for match in matches:
                        match_set = set(match)

                        # Always update atom colors (last alert wins)
                        for atom_idx in match:
                            highlight_atoms.append(atom_idx)
                            highlight_colors[atom_idx] = color

                        # Always update bond colors (last alert wins)
                        for bond in mol.GetBonds():
                            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            if a1 in match_set and a2 in match_set:
                                bidx = bond.GetIdx()
                                highlight_bonds.append(bidx)
                                highlight_bond_colors[bidx] = color

            img = draw_with_colors(
                mol,
                highlight_atoms,
                highlight_colors,
                highlight_bonds,
                highlight_bond_colors,
                size=(300, 300),
            )

            # Figure out group
            pred = int(mol_df.iloc[0]['prediction'])
            label = int(mol_df.iloc[0]['label'])

            if pred == 1 and label == 1:
                correct_toxic_imgs.append((img, mol_id, pred, label))
            elif pred == 0 and label == 0:
                correct_nontoxic_imgs.append((img, mol_id, pred, label))
            else:
                incorrect_imgs.append((img, mol_id, pred, label))

        # Now make 3 separate grids
        plot_group(correct_toxic_imgs, "Correct Toxic (pred=1,label=1)", alert_color_dict, present_alerts)
        plot_group(correct_nontoxic_imgs, "Correct Nontoxic (pred=0,label=0)", alert_color_dict, present_alerts)
        plot_group(incorrect_imgs, "Incorrect Prediction (pred≠label)", alert_color_dict, present_alerts)


    alerts_compiled = load_alerts()







    def get_fragment_smiles(mol, atom_indices):
        """
        Extract the fragment defined by atom_indices from mol
        without canonicalization or atom reordering.
        Assumes atom_indices correspond to one connected component.
        Returns SMILES string or None if extraction fails.
        """
        if mol is None or not atom_indices:
            return None

        n_atoms = mol.GetNumAtoms()
        if any(a >= n_atoms or a < 0 for a in atom_indices):
            raise ValueError(f"Invalid atom index in {atom_indices} (mol has {n_atoms} atoms)")

        # Create a submol by copying only selected atoms and connecting bonds
        atom_indices_set = set(atom_indices)
        emol = Chem.EditableMol(Chem.Mol())

        # Map from old to new atom indices
        old_to_new = {}
        for old_idx in atom_indices:
            old_atom = mol.GetAtomWithIdx(old_idx)
            new_atom = Chem.Atom(old_atom.GetSymbol())
            new_idx = emol.AddAtom(new_atom)
            old_to_new[old_idx] = new_idx

        # Add bonds that connect selected atoms
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in atom_indices_set and a2 in atom_indices_set:
                emol.AddBond(old_to_new[a1], old_to_new[a2], bond.GetBondType())

        submol = emol.GetMol()
        try:
            # Sanitize lightly (skip valence enforcement to avoid issues with radicals)
            Chem.SanitizeMol(submol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
            smi = Chem.MolToSmiles(submol, canonical=False)
            return smi
        except Exception as e:
            print(f"[get_fragment_smiles] Extraction failed: {e}")
            return None

    def compare_fragment_to_alerts(frag_smiles, alerts_compiled):
        if frag_smiles is None:
            return []
        # frag_mol = Chem.MolFromSmiles(frag_smiles)
        try:
            frag_base = Chem.MolFromSmiles(frag_smiles)
            if frag_base is None:
                # Try sanitization: sometimes fragments miss valence info
                try:
                    frag_base = Chem.MolFromSmiles(frag_smiles, sanitize=False)
                    Chem.SanitizeMol(frag_base, catchErrors=True)
                except Exception:
                    print(f"[Warning] Could not sanitize invalid fragment: {frag_smiles}")
                    return []
            frag_mol = Chem.AddHs(frag_base)
        except Exception as e:
            print(f"[Error] Could not parse fragment SMILES {frag_smiles}: {e}")
            return []

        if frag_mol is None:
            return []

        out = []
        frag_natoms = frag_mol.GetNumAtoms()
        for alert_name, patt in alerts_compiled:
            try:
                # Does fragment contain the alert?
                if frag_mol.HasSubstructMatch(patt):
                    # fraction of fragment atoms covered by the alert (take largest mapping)
                    matches = frag_mol.GetSubstructMatches(patt)
                    best = max(matches, key=lambda x: len(x)) if matches else ()
                    overlap_frac = len(best) / float(frag_natoms) if frag_natoms > 0 else 0.0
                    out.append((alert_name, "alert_in_fragment", overlap_frac))
                    continue
                # Does alert contain fragment? (alert mol from patt)
                alert_m = patt  # patt is MolFromSmarts
                if alert_m is not None and alert_m.HasSubstructMatch(frag_mol):
                    # compute fraction of alert atoms matched in fragment (best-effort)
                    matches = alert_m.GetSubstructMatches(frag_mol)
                    best = max(matches, key=lambda x: len(x)) if matches else ()
                    overlap_frac = len(best) / float(alert_m.GetNumAtoms()) if alert_m.GetNumAtoms() > 0 else 0.0
                    out.append((alert_name, "fragment_in_alert", overlap_frac))
                    continue
                # Partial overlap: test if any common substructure by trying matching the fragment into the whole molecule
                # Partial overlap detection is inherently approximate here since we only have the fragment and pattern; we will skip heavy computations
                # caller can compute more exact overlap using compute_overlap_score on full molecule if desired.
            except Exception:
                continue
        return out

    def build_fragment_catalog(per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled,
                               min_pos_count=3, top_k=100):
        n_tasks = len(per_task_impatoms)
        frag_counts_per_task = [Counter() for _ in range(n_tasks)]
        frag_pos_counts_per_task = [Counter() for _ in range(n_tasks)]
        frag_examples = defaultdict(set)
        # global_smiles is a list indexed by mol_id
        n_mols = len(global_smiles)

        for task_idx in range(n_tasks):
            # per_task_impatoms[task_idx] is {task_idx: important_atoms_per_mol}
            imp_for_task = per_task_impatoms[task_idx].get(task_idx, [])
            preds_for_task = per_task_preds[task_idx].get(task_idx, [])
            labels_for_task = per_task_labels[task_idx].get(task_idx, [])

            for mol_id, impdict in enumerate(imp_for_task):
                if mol_id >= n_mols:
                    break
                smiles = global_smiles[mol_id]
                # mol = Chem.MolFromSmiles(smiles) if smiles is not None else None
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles)) if smiles is not None else None
                # combine tight and loose fragments separately: mark their source
                for band in ("tight", "loose"):
                    atom_list = impdict.get(band, [])
                    if not atom_list:
                        continue
                    # Note: atom_list may already be a connected component or set of important atoms.
                    frag_smi = get_fragment_smiles(mol, atom_list)
                    if not frag_smi:
                        continue
                    frag_counts_per_task[task_idx][frag_smi] += 1
                    # If model predicted positive on this mol for this task, count as pos occurrence
                    try:
                        pred = int(preds_for_task[mol_id])
                    except Exception:
                        pred = 1
                    if pred == 1:
                        frag_pos_counts_per_task[task_idx][frag_smi] += 1
                    frag_examples[frag_smi].add(mol_id)

        # Build rows
        all_frags = set()
        for c in frag_counts_per_task:
            all_frags.update(c.keys())

        df_rows = []
        for frag in all_frags:
            counts = [frag_counts_per_task[t][frag] for t in range(n_tasks)]
            pos_counts = [frag_pos_counts_per_task[t][frag] for t in range(n_tasks)]
            total = sum(counts)
            total_pos = sum(pos_counts)
            frag_alert_matches = compare_fragment_to_alerts(frag, alerts_compiled)
            matched_alerts = [m[0] for m in frag_alert_matches] if frag_alert_matches else []
            df_rows.append({
                "fragment": frag,
                "total_count": total,
                "total_pos_count": total_pos,
                **{f"count_t{t}": counts[t] for t in range(n_tasks)},
                **{f"pos_t{t}": pos_counts[t] for t in range(n_tasks)},
                "matched_alerts": ";".join(sorted(set(matched_alerts))),
                "examples": ";".join(str(x) for x in sorted(list(frag_examples.get(frag, set())))[:10])
            })

        # Top-k per task sets:
        per_task_top_sets = []
        for t in range(n_tasks):
            topk = [f for f, _ in frag_counts_per_task[t].most_common(top_k)]
            per_task_top_sets.append(set(topk))

        return df_rows, per_task_top_sets, frag_examples

    def jaccard_matrix(sets_list):
        n = len(sets_list)
        import numpy as np
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                A = sets_list[i];
                B = sets_list[j]
                if not A and not B:
                    M[i, j] = 0.0
                else:
                    M[i, j] = len(A & B) / float(len(A | B))
        return M

    def save_fragment_artifacts(df_rows, per_task_sets, frag_examples, output_dir, alerts_compiled, topN_grid=24):
        # Save CSV summary
        df_frags = pd.DataFrame(df_rows)
        frag_csv = os.path.join(output_dir, "explainer_discovered_fragments_summary.csv")
        df_frags.sort_values("total_pos_count", ascending=False).to_csv(frag_csv, index=False)

        # Identify novel candidates: frequent in positives and not matching known alerts
        df_sorted = df_frags.sort_values("total_pos_count", ascending=False)
        novel = df_sorted[(df_sorted["total_pos_count"] >= min(3, max(3, int(len(global_smiles) * 0.005)))) & (
                df_sorted["matched_alerts"] == "")]
        novel.to_csv(os.path.join(output_dir, "explainer_novel_fragment_candidates.csv"), index=False)

        # Jaccard matrix for top sets
        M = jaccard_matrix(per_task_sets)
        pd.DataFrame(M, index=[f"t{t}" for t in range(len(per_task_sets))],
                     columns=[f"t{t}" for t in range(len(per_task_sets))]).to_csv(
            os.path.join(output_dir, "fragment_topk_jaccard.csv"), index=True)
        # also save heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(6, 5))
            sns.heatmap(M, annot=True, fmt=".2f", xticklabels=[f"t{t}" for t in range(len(per_task_sets))],
                        yticklabels=[f"t{t}" for t in range(len(per_task_sets))], cmap="viridis")
            plt.title("Jaccard of top fragments between tasks")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fragment_topk_jaccard_heatmap.png"))
            plt.close()
        except Exception:
            pass

        # Save grid image of top fragments (top by total_pos_count)
        try:
            top_frags = df_sorted.head(topN_grid)["fragment"].tolist()
            # mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in top_frags]
            mols = [Chem.MolFromSmiles(s) for s in top_frags]
            legends = []
            for _, r in df_sorted.head(topN_grid).iterrows():
                legends.append(f"pos:{r['total_pos_count']},tot:{r['total_count']}")
            img = Draw.MolsToGridImage(mols, molsPerRow=min(6, len(mols)), subImgSize=(200, 200), legends=legends)
            img.save(os.path.join(output_dir, "top_discovered_fragments_grid.png"),dpi=300)
        except Exception:
            pass

        # Create a small JSON mapping for manual inspection (frag -> example mol ids)
        small_map = {r["fragment"]: {"examples": r["examples"], "matched_alerts": r["matched_alerts"],
                                     "total_pos_count": int(r["total_pos_count"])} for r in df_rows}
        with open(os.path.join(output_dir, "fragment_catalog.json"), "w") as fh:
            json.dump(small_map, fh, indent=2)




    try:
        df_rows, per_task_top_sets, frag_examples = build_fragment_catalog(per_task_impatoms, per_task_preds,
                                                                           per_task_labels, global_smiles,
                                                                           alerts_compiled, min_pos_count=3, top_k=200)
        save_fragment_artifacts(df_rows, per_task_top_sets, frag_examples, args.output_dir, alerts_compiled,
                                topN_grid=24)
        print("Fragment-discovery analysis saved to:", args.output_dir)
    except Exception as e:
        print("Fragment-discovery analysis failed:", e)


    def filter_nontrivial_fragments_from_dfrows(df_rows, min_heavy_atoms=4, keep_alert_matches=True):
        """
        From df_rows (list of dicts produced by build_fragment_catalog),
        return deduplicated canonical fragment SMILES that:
          - have at least min_heavy_atoms heavy atoms
          - are not purely hydrocarbon
        If keep_alert_matches==False => also remove fragments that match any known alert (use compare_fragment_to_alerts).
        """
        seen = set()
        kept = []
        for r in df_rows:
            frag = r["fragment"]
            if frag is None or frag == "":
                continue
            m = Chem.MolFromSmiles(frag)
            if m is None:
                # try a non-sanitized parse
                try:
                    m = Chem.MolFromSmiles(frag, sanitize=False)
                    Chem.SanitizeMol(m, catchErrors=True)
                except Exception:
                    continue
            if m is None:
                continue
            if m.GetNumHeavyAtoms() < min_heavy_atoms:
                continue
            if all(a.GetSymbol() in ("C", "H") for a in m.GetAtoms()):
                continue
            # If we must remove alert-matching fragments (for novel set), test:
            if not keep_alert_matches:
                matches = compare_fragment_to_alerts(frag, alerts_compiled)
                if matches:
                    continue
            # deduplicate by canonical SMILES
            can = Chem.MolToSmiles(m, canonical=True)
            if can in seen:
                continue
            seen.add(can)
            kept.append(can)
        return kept

    def cluster_fragments(frag_smiles_list, sim_threshold=0.8):
        """Simple greedy clustering by Tanimoto on Morgan fingerprints. Returns list of (rep, members)."""
        fps = []
        for s in frag_smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            fps.append((s, fp))
        clusters = []  # list of [rep_smi, rep_fp, members_list]
        for s, fp in fps:
            placed = False
            for cluster in clusters:
                rep, rep_fp, members = cluster
                if TanimotoSimilarity(fp, rep_fp) >= sim_threshold:
                    members.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s, fp, [s]])
        # Convert to (representative, members) with counts
        return [(c[0], c[2]) for c in clusters]

    def plot_fragment_clusters(clusters, title, out_path, top_n=20, mols_per_row=5):
        counts = [(rep, len(members)) for rep, members in clusters]
        counts.sort(key=lambda x: x[1], reverse=True)
        top = counts[:top_n]
        mols = [Chem.MolFromSmiles(rep) for rep, _ in top if Chem.MolFromSmiles(rep)]
        legends = [f"{rep}\n({cnt} variants)" for rep, cnt in top]
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=min(mols_per_row, len(mols)), subImgSize=(200, 200),
                                       legends=legends)
            try:
                img.save(out_path)
            except Exception:
                # fallback: show (if running interactive)
                pass
        # also save a CSV of cluster counts
        dfc = pd.DataFrame([{"rep": rep, "n_variants": cnt} for rep, cnt in counts])
        dfc.to_csv(out_path.replace(".png", ".csv"), index=False)
        return dfc

    def compute_top_fragment_task_matrix(df_rows, top_k=50):
        """
        Build matrix indicating which tasks detected which top fragments.
        df_rows: list of dicts with 'fragment' and per-task count_t{t}.
        per_task_counts: optional mapping (not used here).
        Returns:
          - top_frags: list of canonical smiles (top by total_pos_count)
          - matrix: pandas DataFrame rows=fragments, cols=tasks, values=counts (or binary presence)
        """
        df = pd.DataFrame(df_rows)
        if df.empty:
            return [], pd.DataFrame()
        df_sorted = df.sort_values("total_count", ascending=False)
        top = df_sorted.head(top_k)["fragment"].tolist()
        # Build matrix
        task_cols = [c for c in df.columns if c.startswith("count_t")]
        data = []
        for frag in top:
            row = {"fragment": frag}
            sub = df[df["fragment"] == frag]
            if sub.empty:
                # zeros
                for tcol in task_cols:
                    row[tcol] = 0
            else:
                for tcol in task_cols:
                    row[tcol] = int(sub.iloc[0].get(tcol, 0))
            data.append(row)
        mat = pd.DataFrame(data).set_index("fragment")
        # Optionally convert count -> binary presence
        mat_bin = (mat > 0).astype(int)
        return top, mat, mat_bin



    # ---------- Use existing df_rows and frag_examples from build_fragment_catalog ----------
    # df_rows is list of dicts OR DataFrame; ensure it's a DataFrame
    if isinstance(df_rows, list):
        df_all_frags = pd.DataFrame(df_rows)
    else:
        df_all_frags = df_rows.copy()

    # === START: Combined plot of known vs novel fragments ===

    def get_fragment_info_lists(df_rows, alerts_compiled, min_heavy_atoms=2):
        """Split fragments into alert-matched and novel lists, with metadata for plotting."""
        alert_frags, novel_frags = [], []
        for r in df_rows:
            frag = r["fragment"]
            if not frag:
                continue
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue
            if mol.GetNumHeavyAtoms() < min_heavy_atoms:
                continue
            if all(a.GetSymbol() in ("C", "H") for a in mol.GetAtoms()):
                continue

            # See if it matches alerts
            matches = compare_fragment_to_alerts(frag, alerts_compiled)
            alerts = [m[0] for m in matches] if matches else []
            entry = {
                "fragment": frag,
                "mol": mol,
                "alerts": alerts,
                "total_pos_count": r.get("total_pos_count", 0),
                "total_count": r.get("total_count", 0),
            }
            if alerts:
                alert_frags.append(entry)
            else:
                novel_frags.append(entry)

        # sort by total_pos_count descending
        alert_frags.sort(key=lambda x: x["total_pos_count"], reverse=True)
        novel_frags.sort(key=lambda x: x["total_pos_count"], reverse=True)
        return alert_frags, novel_frags

    def plot_combined_known_vs_novel(alert_frags, novel_frags, out_path, top_n_each=12):
        """Create a single grid image showing known (alert) and novel fragments grouped."""
        n_show_alert = min(len(alert_frags), top_n_each)
        n_show_novel = min(len(novel_frags), top_n_each)
        show_list = alert_frags[:n_show_alert] + novel_frags[:n_show_novel]

        mols = [e["mol"] for e in show_list]
        legends = []
        for e in show_list:
            smi = e["fragment"]
            if e["alerts"]:
                alerts_str = ", ".join(e["alerts"])
                legends.append(f"{smi}\n(alerts: {alerts_str})")
            else:
                legends.append(f"{smi}\n(novel)")
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=6,
            subImgSize=(220, 220),
            legends=legends,
            useSVG=False,
        )
        try:
            img.save(out_path)
        except Exception:
            pass
        return n_show_alert, n_show_novel

    # Compute fragment sets and plot
    alert_frags, novel_frags = get_fragment_info_lists(df_rows, alerts_compiled, min_heavy_atoms=4)
    out_combined_img = os.path.join(args.output_dir, "fragments_known_vs_novel_combined.png")
    n_alert, n_novel = plot_combined_known_vs_novel(alert_frags, novel_frags, out_combined_img, top_n_each=12)
    print(
        f"Saved combined fragment grid with {n_alert} known (alert-matched) and {n_novel} novel fragments -> {out_combined_img}")

    # === END: Combined plot of known vs novel fragments ===

    assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, args.output_dir, alerts_list)


if __name__ == "__main__":
    main()

    