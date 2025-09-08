from datetime import datetime
import faulthandler
import os
import pdb
import re
import sys
import csv
import h5py
import random
import markdown
import csv
import argparse
import logging

import pickle
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboard.compat.proto.struct_pb2 import NoneValue
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import GNNExplainer, PGExplainer, Explainer
from torch_geometric.utils import to_networkx
import networkx as nx
from networkx.drawing import nx_agraph
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops
from torch_geometric.utils import to_networkx
from collections import Counter
from collections import defaultdict
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import math
import matplotlib.cm as cm
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image, ImageDraw, ImageFont
import colorsys

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


    compiled = []

    for name, smarts in alerts:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            compiled.append((name, patt))
    return compiled

# Compute overlap between substructure matches and important atoms
def compute_overlap_score(mol, smarts, highlighted_atoms):
    matches = mol.GetSubstructMatches(smarts) #match_smarts(mol, smarts)
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

    for i, (smiles, imp_dict, pred, label, label_overall) in enumerate(zip(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall)):
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
                     highlight_bonds, highlight_bond_colors, size=(300,300)):
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
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

def blank_image(size=(300,300), color=(255,255,255)):
    return Image.new('RGB', size, color)

def hstack_images(imgs, pad=6, bg=(255,255,255)):
    widths, heights = zip(*(i.size for i in imgs))
    total_w = sum(widths) + pad*(len(imgs)-1)
    max_h = max(heights)
    new_im = Image.new('RGB', (total_w, max_h), bg)
    x = 0
    for im in imgs:
        new_im.paste(im, (x, (max_h-im.size[1])//2))
        x += im.size[0] + pad
    return new_im

import matplotlib.cm as cm
import numpy as np

def generate_alert_colors(alerts):
    reserved = [
        (1.0, 0.3, 0.3),   # red
        (1.0, 0.8, 0.3),   # orange
        (0.8, 0.8, 0.8),   # gray
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
        if np.all(dists > 0.25):   # threshold to avoid looking too similar
            safe_colors.append(cand)
        else:
            # tweak hue slightly if too close
            new = ((cand[0]*0.7 + 0.3), (cand[1]*0.7), (cand[2]*0.7))
            safe_colors.append(new)

    return {name: safe_colors[i] for i,(name,_) in enumerate(alerts)}


# Plot summary
def assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, output_dir):
    #cmap = cm.get_cmap("tab20", len(alerts_compiled))
    #alert_colors = {name: tuple(cmap(i)[:3]) for i, (name, _) in enumerate(alerts_compiled)}

    alert_colors = generate_alert_colors(alerts_compiled)

    os.makedirs(os.path.join(output_dir, 'summary_rows'), exist_ok=True)

    n_tasks = len(per_task_dfs)
    n_mols = len(global_smiles)
    cell_size = (300,300)

    alerts_present_by_mol = []
    for s in global_smiles:
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
                            bond_highlights[bond.GetIdx()] = (0.6,0.6,0.6)
        alerts_present_by_mol.append((present, atom_highlights, bond_highlights))

    all_row_imgs = []
    rows_correct_toxic = []
    rows_correct_nontoxic = []
    rows_incorrect = []

    for mol_id in range(n_mols):
        df0 = per_task_dfs[0]
        df_0 = df0[0]
        overall_rows = df_0[df_0['mol_id']==mol_id]
        if overall_rows.empty:
            continue
        overall_label = int(overall_rows.iloc[0]['label_overall'])
        if overall_label == -1:
            continue

        mol = Chem.MolFromSmiles(global_smiles[mol_id])

        strain_cells = []
        for task in range(n_tasks):
            pdf = per_task_dfs[task]
            pdf_task = pdf[task]
            mol_df = pdf_task[pdf_task['mol_id']==mol_id]
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
                        patt = next((p for n,p in alerts_compiled if n==name), None)
                        if patt is None:
                            continue
                        matches = mol.GetSubstructMatches(patt)
                        color = alert_colors.get(name, (0.5,0.5,0.5))
                        for match in matches:
                            for a in match:
                                highlight_atoms.append(a)
                                atom_colors[a] = color
                            for bond in mol.GetBonds():
                                a1,a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
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
                draw.rectangle([(0,0),(im.size[0],18)], fill=(255,255,255))
                draw.text((4,0), text, fill=(0,0,0), font=font)
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
        d.text((10,10), f"Consensus: {consensus}", fill=(0,0,0), font=font)
        d.text((10,40), f"Overall label: {overall_label}", fill=(0,0,0), font=font)

        present, atom_highlights, bond_highlights = alerts_present_by_mol[mol_id]
        im_alerts_present = draw_with_colors(mol, list(atom_highlights.keys()), atom_highlights, list(bond_highlights.keys()), bond_highlights, size=cell_size)

        union_atoms, atom_cols = set(), {}
        for t in range(n_tasks):
            imp = per_task_impatoms[t]
            imp_t = imp[t]
            tight = imp_t[mol_id]['tight'] if mol_id < len(imp_t) else []
            loose = imp_t[mol_id]['loose'] if mol_id < len(imp_t) else []
            for a in tight:
                union_atoms.add(a)
                atom_cols[a] = (1.0, 0.3, 0.3)
            for a in loose:
                if a not in atom_cols:
                    atom_cols[a] = (1.0, 0.8, 0.3)
                union_atoms.add(a)
        im_imp = draw_with_colors(mol, list(union_atoms), atom_cols, [], {}, size=cell_size)

        row_imgs = strain_cells + [cons_im, im_alerts_present]
        row_concat = hstack_images(row_imgs, pad=4)

        outfn = os.path.join(output_dir, 'summary_rows', f'mol_{mol_id:04d}_cons_{consensus}_ovl_{overall_label}.png')
        #row_concat.save(outfn)
        #all_row_imgs.append(row_concat.convert('RGB'))
        #print(f"Saved summary row for mol {mol_id} -> {outfn}")

        if consensus == 1 and overall_label == 1:
            rows_correct_toxic.append(row_concat)
        elif consensus == 0 and overall_label == 0:
            rows_correct_nontoxic.append(row_concat)
        else:
            rows_incorrect.append(row_concat)

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
        batch = row_imgs[i:i+rows_per_page]
        widths, heights = zip(*(im.size for im in batch))
        page_w = max(widths)
        page_h = sum(heights)
        page = Image.new("RGB", (page_w, page_h), (255,255,255))
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

def make_legend_page(alert_colors, size=(1200,1600)):
    page = Image.new("RGB", size, (255,255,255))
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    x, y = 50, 50
    # structural alerts
    draw.text((x, y), "Structural Alerts:", fill=(0,0,0), font=font)
    y += 40
    for name, color in alert_colors.items():
        rgb = tuple(int(c*255) for c in color)
        draw.rectangle([x,y,x+40,y+40], fill=rgb)
        draw.text((x+50, y), name, fill=(0,0,0), font=font)
        y += 50
        if y > size[1]-50:
            y = 90
            x += 400

    return page



def main():
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_file = args.input_file

    with open( input_file, 'r' ) as input_stream:
        input_data = yaml.load(input_stream, Loader=yaml.Loader)

    # Set database path
    database_path = input_data.get("database", "./GraphDataBase_AMES")

    # The database is described with its own yaml file; so read it
    database_file = database_path + '/graph_description.yml'

    with open( database_file, 'r' ) as database_stream:
        database_data = yaml.load(database_stream, Loader=yaml.Loader)

    # Model parameters
    n_graph_convolution_layers = input_data.get("nGraphConvolutionLayers", 0) # Number of graph convolutional layers
    n_node_neurons = input_data.get("nNodeNeurons", None) # Number of neurons in GNN
    n_edge_neurons = input_data.get("nEdgeNeurons", None) # Number of edges in GNN
    dropout_GNN = input_data.get("dropoutGNN", None) # Dropout GNN
    momentum_batch_norm = input_data.get("momentumBatchNorm", None) # Batch norm GNN

    n_shared_layers = input_data.get("nSharedLayers", 4) # Number of layers in shared core
    n_target_specific_layers = input_data.get("nTargetSpecificLayers", 2) # Number of layers in target specific core
    n_shared = input_data.get("nShared", None) # Number of neurons in shared core
    n_target = input_data.get("nTarget", None)  # Number of neurons in target specific core
    dropout_shared = input_data.get("dropoutShared", None) # Dropout in shared core
    dropout_target = input_data.get("dropoutTarget", None) # Dropout in target specific core

    activation = input_data.get("ActivationFunction", "ReLU") # Activation function
    weighted_loss_function = input_data.get("weightedCostFunction", False)
    w1 = input_data.get("w1", 1.0)
    w2 = input_data.get("w2", 1.0)
    w3 = input_data.get("w3", 1.0)
    w4 = input_data.get("w4", 1.0)
    w5 = input_data.get("w5", 1.0)
    if weighted_loss_function:
        #class_weights = {
        #    '98': {0: 1.0, 1: w1, -1: 0},
        #    '100': {0: 1.0, 1: w2, -1: 0},
        #    '102': {0: 1.0, 1: w3, -1: 0},
        #    '1535': {0: 1.0, 1: w4, -1: 0},
        #    '1537': {0: 1.0, 1: w5, -1: 0},
        #}
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
    if bond_angle_features: n_edge_features += 1 # bond-angle feature
    if dihedral_angle_features: n_edge_features += 1 # dihedral-angle feature

    # Training parameters
    nEpochs = input_data.get("nEpochs", 10) # Number of epochs
    nBatch = input_data.get("nBatch", 50) # Batch size
    chkptFreq = input_data.get("nCheckpoint", 10) # Checkpoint frequency
    seed = input_data.get("randomSeed", 42) # Random seed
    nTrainMaxEntries = input_data.get("nTrainMaxEntries", None) # Number of training examples to use (if not using whole dataset)
    nValMaxEntries = input_data.get("nValMaxEntries", None) # Number of validation examples to use (if not using whole dataset)
    learningRate = input_data.get("learningRate", 0.0001) # Learning rate
    weightedCostFunction = input_data.get("weightedCostFunction", None) # Use weighted  cost function
    L2Regularization = input_data.get("L2Regularization", 0.005) # L2 regularization coefficient
    loadModel = input_data.get("loadModel", False)
    loadOptimizer = input_data.get("loadOptimizer", False)
    useMolecularDescriptors = input_data.get("useMolecularDescriptors", False) # Use molecular descriptors instead of graphs for comparison to original MTL paper

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
    model = BuildNN_GNN_MTL(n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features, n_edge_features, dropout_GNN, momentum_batch_norm,
                            n_shared_layers, n_target_specific_layers, n_shared, n_target, dropout_shared, dropout_target,
                            activation, useMolecularDescriptors, n_inputs)

    checkpoint = torch.load('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output_seed_8_20_25/checkpoints/metrics_45_1.pt', map_location=torch.device('cpu'))
    
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
        model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)

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
                #global_feats=data.global_feats
            )

            with torch.no_grad():
                task_output = task_model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch,
                    #global_feats=data.global_feats
                )

                prediction = int(task_output.item() > 0.5)  # 1 = toxic, 0 = non-toxic
                predictions.append(prediction)

            edge_mask = explanation.edge_mask.detach().cpu().numpy()

            # Tight filter
            k_edges_tight = max(8, int(0.15 * edge_mask.size))  # ~10–15%
            top_e_tight = np.argsort(-edge_mask)[:k_edges_tight]

            imp_edges_tight = data.edge_index[:, torch.tensor(top_e_tight, device=data.edge_index.device)]
            imp_nodes_tight = sorted(set(imp_edges_tight.view(-1).tolist()))

            G = to_networkx(data, to_undirected=True)
            sub_tight = G.subgraph(imp_nodes_tight).copy()
            if sub_tight.number_of_nodes() > 0:
                lcc_tight = max(nx.connected_components(sub_tight), key=len)
                important_atoms_tight = sorted(list(lcc_tight))
            else:
                important_atoms_tight = []

            # Loose filter
            k_edges_loose = max(8, int(0.15 * edge_mask.size))  # ~25–30%
            top_e_loose = np.argsort(-edge_mask)[:k_edges_loose]

            imp_edges_loose = data.edge_index[:, torch.tensor(top_e_loose, device=data.edge_index.device)]
            imp_nodes_loose = sorted(set(imp_edges_loose.view(-1).tolist()))

            sub_loose = G.subgraph(imp_nodes_loose).copy()
            if sub_loose.number_of_nodes() > 0:
                # Keep *all* connected components, not just the largest
                comps = max(nx.connected_components(sub_loose), key=len)
                important_atoms_loose = sorted(list(comps))
                #comps = list(nx.connected_components(sub_loose))
                #important_atoms_loose = sorted(set().union(*comps))
            else:
                important_atoms_loose = []

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

        df = evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, correct_val, correct_val_overall) # Compute overlap scores by comparing alerts and important nodes, store all in df
        per_task_dfs.append({task_id: df})

        summary = df.groupby("alert")[["tight_score", "loose_score"]].mean().reset_index() #compute average scores for each alert
        summary = summary.sort_values("loose_score", ascending=False) # sort by loose score
        alerts_csv = os.path.join(args.output_dir, f"alerts_task_{task}.csv") # write to csv
        summary.to_csv(alerts_csv, index=False)

        ##### Plot per-strain results
        plt.figure(figsize=(10, 6))
        plt.bar(summary["alert"], summary["tight_score"], alpha=0.6)
        #plt.bar(summary["alert"], summary["loose_score"], alpha=0.6, label="Loose")
        plt.xticks(rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title("GNN explanation overlap with functional alerts, all molecules")
        plt.legend()
        plt.tight_layout()
        plt.show()

        alerts_list = df['alert'].unique()
        x = np.arange(len(alerts_list))
        bar_width = 0.5  # width of the bars

        # --- Tight ---
        tight_means = df.groupby(["alert", "prediction"])["tight_score"].mean().unstack(fill_value=0)

        mean_tight_nontoxic = tight_means.get(0, np.zeros(len(alerts_list))).mean()
        mean_tight_toxic = tight_means.get(1, np.zeros(len(alerts_list))).mean()

        plt.figure(figsize=(12, 6))
        plt.bar(x, tight_means.get(0, np.zeros(len(alerts_list))), width=bar_width,
                color='purple', alpha=0.6, label=f"Nontoxic (pred=0), overall mean={mean_tight_nontoxic:.2f}")
        plt.bar(x, tight_means.get(1, np.zeros(len(alerts_list))), width=bar_width,
                color='red', alpha=0.6, label=f"Toxic (pred=1), overall mean={mean_tight_toxic:.2f}")
        plt.xticks(x, alerts_list, rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title(f"Task {task}: Mean overlap score by predicted toxicity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # --- Loose ---
        loose_means = df.groupby(["alert", "prediction"])["loose_score"].mean().unstack(fill_value=0)

        mean_loose_nontoxic = loose_means.get(0, np.zeros(len(alerts_list))).mean()
        mean_loose_toxic = loose_means.get(1, np.zeros(len(alerts_list))).mean()

        plt.figure(figsize=(12, 6))
        plt.bar(x, loose_means.get(0, np.zeros(len(alerts_list))), width=bar_width,
                color='purple', alpha=0.6, label=f"Nontoxic (pred=0), overall mean={mean_loose_nontoxic:.2f}")
        plt.bar(x, loose_means.get(1, np.zeros(len(alerts_list))), width=bar_width,
                color='red', alpha=0.6, label=f"Toxic (pred=1), overall mean={mean_loose_toxic:.2f}")
        plt.xticks(x, alerts_list, rotation=90)
        plt.ylabel("Mean overlap score")
        plt.title(f"Task {task}: Mean overlap score by predicted toxicity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        pivot_df = df.pivot(index="mol_id", columns="alert", values="loose_score")

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, cmap="viridis", cbar_kws={"label": "Overlap score"})
        plt.title("Per-molecule overlap with functional alerts (loose)")
        plt.xlabel("Alert")
        plt.ylabel("Molecule")
        plt.tight_layout()
        plt.show()

        # Consider an alert present if tight_score > 0 or loose_score > 0
        df['alert_present'] = (df['tight_score'] > 0) | (df['loose_score'] > 0)
        #df['alert_present'] = (df['tight_score'] > 0)

        alert_summary = df.groupby("alert").agg(
            n_positive_predictions=("prediction", lambda x: ((x == 1) & df.loc[x.index, 'alert_present']).sum()),
            n_negative_predictions=("prediction", lambda x: ((x == 0) & df.loc[x.index, 'alert_present']).sum()),
            n_positive_labels=("label", lambda x: ((x == 1) & df.loc[x.index, 'alert_present']).sum()),
            n_negative_labels=("label", lambda x: ((x == 0) & df.loc[x.index, 'alert_present']).sum()),
        ).reset_index()

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
        valid_mols = df[(df['label_overall'] != -1) & (df['label'] != -1)]['mol_id'].unique()

        # Use tab20 for distinct alert colors
        alert_names = df['alert'].unique()
        n_alerts = len(alert_names)
        cmap = cm.get_cmap('tab20', n_alerts)
        alert_color_dict = {alert_names[i]: cmap(i)[:3] for i in range(n_alerts)}
        alert_color_dict['Heterocyclic/polycyclic aromatic hydrocarbons'] = (0.8,0.0,0.8)
        present_alerts = set()

        correct_toxic_imgs = []
        correct_nontoxic_imgs = []
        incorrect_imgs = []

        for mol_id in valid_mols:
            #mol_df = df[(df['mol_id'] == mol_id) & (df['alert_present'])]
            mol_df = df[(df['mol_id'] == mol_id)]
            mol = Chem.MolFromSmiles(smiles_list[mol_id])
            #if mol is None or mol_df.empty:
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
    assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, args.output_dir)

if __name__ == "__main__":
    main()

    