# Consolidated from GNN_explainer_analysis_final.py and GNN_explainer_analysis_input_features.py
from datetime import datetime
import faulthandler
import os
import io
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
from matplotlib.lines import Line2D
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
    parser.add_argument("--checkpoint_file", type=str, required=True,
                        help="Path to .pt model checkpoint file")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to data.csv; if not set, uses data_file from YAML")
    parser.add_argument("--analyze_input_features", action="store_true",
                        help="If set, also run Integrated Gradients input feature importance analysis")
    parser.add_argument("--analyze_input_features_only", action="store_true",
                        help="Run only Integrated Gradients analysis; skip GNNExplainer entirely")
    return parser.parse_args()

# Load structural alerts as SMARTS
def load_alerts():
    # Updated
    alerts = [
        ("Acyl halides", "[Br,Cl,F,I][CX3](=[OX1])[#1,*&!$([OH1])&!$([SH1])]"),
        ("Alkyl and aryl N-nitroso groups", "[#6][NX3][NX2]=[OX1]"),
        ("Alkyl or benzyl esters of phosphonic or sulphonic acids","[$([Sv6X4;!$([Sv6X4][OH]);!$([Sv6X4][SH]);!$([Sv6X4][O-]);!$([Sv6X4][S-])](=[OX1])(=[OX1])[$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)]),$([Pv5X4;!$([Pv5X4][OH]);!$([Pv5X4][SH]);!$([Pv5X4][O-]);!$([Pv5X4][S-])](=[OX1])([$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)])[$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)])]"),
        ("Alkyl carbamate and thiocarbamates", "[NX3]([C,#1])([C,#1])[CX3](=[OX1,Sv2X1])[OX2,Sv2X2]C"),
        ("Hydrazines", "[NX3;!$([NX3](=[OX1])=[OX1]);!$([NX3+](=[OX1])[O-])][NX3;!$([NX3](=[OX1])=[OX1]);!$([NX3+](=[OX1])[O-])]"),
        ("Alkyl nitrites", "[OX1]=[NX2][OX2][CX4]"),
        ("Aliphatic azo and azoxy groups", "[$([C,#1][NX2]=[NX2][C,#1]),$([CX3]=[NX2+]=N),$([CX3]=[NX2+]=[NX1-]),$([CX3-][NX2+]#[NX1]),$([CX3][NX2]#[NX1]),$(C[NX2]=N(=O)[*]),$(C[NX2]=[N+]([O-])[*])]"),
        ("Aliphatic halogens", "[CX4;!H0][Br,Cl,I]"),
        ("Aliphatic N-nitro groups", "[NX3]([#1,C])([#1,C])[$([NX3+](=[OX1])[O-]),$([NX3](=O)=O)]"),
        ("Alpha, beta unsaturated aliphatic alkoxy groups", "C[CX3;H1]=[CX3;H1][OX2][#6]"),
        ("Alpha, beta unsaturated carbonyls", " [CX3]([!$([OH]);!$([O-])])(=[OX1])[CX3H1]=[CX3]([$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH3]),$([CH]([CH3])[CH3]),$([CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH3]),$([CH2][CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH3]),$([CH2][CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH2][CH3]),$([CH2][CH]([CH3])[CH2][CH3]),$([CH2][CH2][CH]([CH3])[CH3]),$([CH]([CH2][CH3])[CH2][CH3]),$([CH]([CH3])[CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH2][CH3]),$([CH2][CH0]([CH3])([CH3])[CH3]),$([#1,#7,#8,F,Cl,Br,I,#15,#16,#5]),$([CH]=[CH][#6]);!$([a!r0])])[$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH3]),$([CH]([CH3])[CH3]),$([CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH3]),$([CH2][CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH3]),$([CH2][CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH2][CH3]),$([CH2][CH]([CH3])[CH2][CH3]),$([CH2][CH2][CH]([CH3])[CH3]),$([CH]([CH2][CH3])[CH2][CH3]),$([CH]([CH3])[CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH2][CH3]),$([CH2][CH0]([CH3])([CH3])[CH3]),$([#1,#7,#8,F,Cl,Br,I,#15,#16,#5]),$([CH]=[CH][#6]);!$([a!r0])]"),
        ("Aromatic amines and hydroxylamines", "[a!r0][$([NH2]),$([NX3][OX2H1]),$([NX3][OX2][CX3H1](=[OX1])),$([NX2]=[CH2]),$([NX2]=C=[OX1]);!$([NX3,NX2]a(a-[!#1])a-[!#1]);!$([NX3,NX2]aa-C(=[OX1])[OH]);!$([NX3,NX2]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3,NX2]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3,NX2]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic diazo groups", "[$([NX2]([a!r0])=[NX2][a!r0]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH])!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH]);!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH])]"),
        ("Aromatic nitro groups", "[a!r0][$([NX3+](=[OX1])[O-]),$([NX3](=[OX1])=[OX1]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic nitroso groups", "[a!r0][NX2]=[OX1]"),
        ("Aromatic mono- and dialkyl amino groups", "[a!r0][$([NX3;H1][CH3]),$([NX3;H1][CH2][CH3]),$([NX3]([CH3])[CH3]),$([NX3]([CH3])[CH2][CH3]),$([NX3]([CH2][CH3])[CH2][CH3]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic N-acyl amines", "[a!r0][$([NX3;H1]),$([NX3][CH3]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])][CX3](=[OX1])([$([#1]),$([CH3])])"),
        ("Aromatic N-oxides", "[O-][N+]1=CC=CC=C1"),
        ("Azide and triazene groups", "[$([NX2!R]=[NX2!R][NX3!R]),$([NX2]=[NX2+]=[NX1-]),$([NX2]=[NX2+]=N)]"),
        ("Coumarins and Furocoumarins", "[$(c1cccc2c1oc(=O)cc2),$(C1=CC(=O)OC2=CC=CC=C12),$(C1=CC(=O)Oc2ccccc12)]"),
        ("Epoxides and aziridines", "[CX4]1[OX2,NX3][CX4]1"),
        ("Polycyclic aromatic hydrocarbons", "[$([cX3R3]),$([cX3;R1,R2,R3]1[cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R2,R3]2[cX3;R2,R3]1[cX3;R1,R2,R3][cX3;R2,R3]3[cX3;R2,R3]([cX3;R1,R2,R3]2)[cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3]3)].[!$([n,o,s])]"),
        ("Heterocyclic polycyclic aromatic hydrocarbons", "[$([aR3].[n,o,s]),$([$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[n,o,s])]"),
        ("Isocyanate and isothiocyanate groups", "[NX2]=[CX2]=[OX1,Sv2X1]"),
        ("Monohaloalkenes", "[CX3]([CX4,#1])([F,Cl,Br,I])=[CX3]([CX4,#1])[!F!Cl!Br!I]"),
        ("N-methylol derivatives", "[OX2;H1][CH2][NX3]"),
        ("Propiolactones and propiosultones", "[$([OX2]1[CX4][CX4][CX3]1(=[OX1])),$([CX4]1[CX4][CX4][Sv6;X4](=[OX1])(=[OX1])[OX2]1)]"),
        ("Quinones", "[$([#6X3]1=,:[#6X3]-,:[#6X3](=[OX1])-,:[#6X3]=,:[#6X3]-,:[#6X3]1(=[OX1])),$([#6X3]1(=[OX1])-,:[#6X3](=[OX1])-,:[#6X3]=,:[#6X3]-,:[#6X3]=,:[#6X3]1)]"),
        ("Simple aldehydes", "[CX3]([H])(=[OX1])[#1,#6&!$([CX3]=[CX3])]"),
        ("S- or N- mustards", "[F,Cl,Br,I][CH2][CH2][NX3,SX2][CH2][CH2][F,Cl,Br,I]")
    ]

    compiled = []

    for name, smarts in alerts:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            compiled.append((name, patt))
    return compiled


# Extra SMARTS used ONLY by the novel-fragment detection (NOT the overlap/heatmap/AUC analyses), to
# suppress "near-miss" categories the base list lacks so they stop showing up as novel candidates.
_EXTRA_ALERTS = [
    ("Aromatic azo (extended)", "[c][NX2]=[NX2][#6]"),
    ("gem/poly-halo alkane (extended)", "[CX4]([F,Cl,Br,I])[F,Cl,Br,I]"),
    ("1,1-/halo alkene (extended)", "[CX3](=[CX3])[F,Cl,Br,I]"),
    ("Nitro group, any (extended)", "[$([NX3](=O)=O),$([N+](=O)[O-])]"),
    ("Sulfonate/sulfate ester (extended)", "[OX2][SX4](=O)(=O)"),
]


def load_extended_alerts():
    """Base alerts (load_alerts) PLUS a few extra near-miss SMARTS, for the NOVEL-FRAGMENT analysis
    only. load_alerts() is intentionally left unchanged so every other alert analysis is unaffected."""
    compiled = load_alerts()
    for name, smarts in _EXTRA_ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            compiled.append((name, patt))
    return compiled

def expand_fragment_atoms(mol, atom_indices, radius=1):
    expanded = set(atom_indices)
    for a in atom_indices:
        queue = [(a, 0)]
        visited = set()
        while queue:
            idx, dist = queue.pop()
            if dist == radius:
                continue
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                ni = nbr.GetIdx()
                if ni not in expanded:
                    expanded.add(ni)
                if ni not in visited:
                    visited.add(ni)
                    queue.append((ni, dist + 1))
    return sorted(expanded)


def compute_alert_fps(alerts_compiled, fp_radius=2, fp_bits=2048):
    alert_fps = {}
    for alert_name, patt in alerts_compiled:

        # Copy pattern so original query mol stays untouched
        patt_copy = Chem.Mol(patt)

        # Light sanitization to make Morgan FP safe
        try:
            Chem.SanitizeMol(
                patt_copy,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            )
        except Exception:
            pass

        # Generate fingerprint
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            patt_copy, fp_radius, nBits=fp_bits
        )
        alert_fps[alert_name] = fp

    return alert_fps


def expand_match_atoms(mol, match_atoms):
    """Expand SMARTS match atoms to include full rings and direct heavy-atom neighbors."""
    expanded = set(match_atoms)

    # Include all atoms in rings that contain a match atom
    for atom_idx in match_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.IsInRing():
            for ring in mol.GetRingInfo().AtomRings():
                if atom_idx in ring:
                    expanded.update(ring)

    # Include direct heavy-atom neighbors
    neighbors = set()
    for atom_idx in match_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1:
                neighbors.add(neighbor.GetIdx())
    expanded.update(neighbors)

    return expanded


# Compute overlap between substructure matches and important atoms
def compute_overlap_score(mol, smarts, highlighted_atoms):
    matches = mol.GetSubstructMatches(smarts)
    if not matches:
        return 0.0, []

    highlighted_atoms = set(highlighted_atoms)
    scores = []

    for match in matches:
        match_atoms = set(match)
        overlap = len(match_atoms & highlighted_atoms) / len(match_atoms)
        scores.append(overlap)

    return max(scores), matches

# Compute overlap scores, return df
def evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, probs, correct_val, correct_val_overall):
    rows = []

    for i, (smiles, imp_dict, pred, prob, label, label_overall) in enumerate(
            zip(smiles_list, important_atoms_per_mol, predictions, probs, correct_val, correct_val_overall)):
        # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            for name, smarts in alerts:
                rows.append({
                    "mol_id": i, "alert": name,
                    "tight_score": 0, "loose_score": 0,
                    "prediction": pred, "prob": prob,
                    "label": label,
                    "label_overall": label_overall,
                    "alert_matched": False,
                })
            continue
        for name, smarts in alerts:
            tight_score, _ = compute_overlap_score(mol, smarts, imp_dict["tight"])
            loose_score, _ = compute_overlap_score(mol, smarts, imp_dict["loose"])
            alert_matched = bool(mol.GetSubstructMatches(smarts))

            rows.append({
                "mol_id": i,
                "alert": name,
                "tight_score": tight_score,
                "loose_score": loose_score,
                "prediction": pred, "prob": prob,
                "label": label,
                "label_overall": label_overall,
                "alert_matched": alert_matched,
            })

    return pd.DataFrame(rows)


# Get fragments based on GNNExplainer important atoms
def build_fragment_catalog(per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, alert_fps,
                            min_pos_count=3, top_k=100):
    n_tasks = len(per_task_impatoms)
    frag_counts_per_task = [Counter() for _ in range(n_tasks)]
    frag_pos_counts_per_task = [Counter() for _ in range(n_tasks)]
    frag_examples = defaultdict(set)
    n_mols = len(global_smiles)

    for task_idx in range(n_tasks):
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
            atom_list = impdict.get("loose", [])
            if not atom_list:
                continue
            atom_list = expand_fragment_atoms(mol, atom_list, radius=1)
            frag_smi = get_fragment_smiles(mol, atom_list) # Generate submol based on important atoms
            if not frag_smi:
                continue
            frag_counts_per_task[task_idx][frag_smi] += 1
            # If model predicted positive on this mol for this task, count as pos occurrence
            pred = int(preds_for_task[mol_id])
            if pred == 1:
                frag_pos_counts_per_task[task_idx][frag_smi] += 1 # Fragment present in toxic molecule
            frag_examples[frag_smi].add(mol_id) # Overall fragments

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
        frag_alert_matches = compare_fragment_to_alerts(frag, alerts_compiled, alert_fps)
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


# Radii of the circular environments mined around important atoms (tunable).
_SUBSTRUCTURE_RADII = (2, 3)

# Functional groups kept INTACT when cutting circular environments, so the radius boundary never
# truncates them into chemically meaningless bare-[N+]/OS artifacts (which also hid known alerts).
_COMPLETION_SMARTS = [
    "[$([NX3](=O)=O),$([N+](=O)[O-])]",  # nitro (C- or N-)
    "[SX4](=O)(=O)",                      # sulfonyl / sulfonate / sulfate
    "[NX2]=[NX2]",                        # azo / azoxy
    "[NX2]=O",                            # nitroso
    "[CX2]#[NX1]",                        # nitrile
]
_COMPLETION_PATTS = [Chem.MolFromSmarts(s) for s in _COMPLETION_SMARTS]


def _group_matches(mol):
    """Atom-index sets of completion functional groups in `mol`, for boundary completion."""
    out = []
    for patt in _COMPLETION_PATTS:
        if patt is None:
            continue
        for m in mol.GetSubstructMatches(patt):
            out.append(set(m))
    return out


def _circular_env_smiles(mol, atom_idx, radius, group_matches=()):
    """Canonical, H-free SMILES of the radius-`radius` circular environment around `atom_idx`, with
    any touched functional group completed and net-charged/invalid artifacts dropped."""
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env:
        return None
    atoms = set()
    for b in env:
        bond = mol.GetBondWithIdx(b)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    if len(atoms) < 2:
        return None
    # Complete any functional group the environment touches (no truncated nitro/sulfonate/azo/etc.).
    for g in group_matches:
        if atoms & g:
            atoms |= g
    try:
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True)
        fm = Chem.MolFromSmiles(smi, sanitize=False)
        if fm is None:
            return None
        Chem.SanitizeMol(fm, catchErrors=True)
        fm = Chem.RemoveHs(fm)
        if Chem.GetFormalCharge(fm) != 0:   # drop residual bare-[N+]/carbanion/under-valent artifacts
            return None
        return Chem.MolToSmiles(fm) or None
    except Exception:
        return None


def build_substructure_catalog(per_task_impatoms, per_task_preds, per_task_labels, global_smiles,
                               alerts_compiled, alert_fps, radii=_SUBSTRUCTURE_RADII, top_k=200,
                               **_ignored):
    """Recurring-substructure mining for novel-fragment detection (replaces the whole-region dedup).

    For each molecule/task, extracts the radius-r circular environments around the model's TIGHT
    important HEAVY atoms (on the implicit-H mol; heavy indices are graph-aligned, so this also keeps
    hydrogens out of the novel analysis), deduplicates substructures within the molecule, and tallies
    occurrence/positive-occurrence counts across all explanations. Same `df_rows` schema as
    build_fragment_catalog, so all downstream fragment functions/figures are unchanged.
    """
    n_tasks = len(per_task_impatoms)
    n_mols = len(global_smiles)
    frag_counts_per_task = [Counter() for _ in range(n_tasks)]
    frag_pos_counts_per_task = [Counter() for _ in range(n_tasks)]
    frag_examples = defaultdict(set)

    for task_idx in range(n_tasks):
        imp_for_task = per_task_impatoms[task_idx].get(task_idx, [])
        preds_for_task = per_task_preds[task_idx].get(task_idx, [])
        for mol_id, impdict in enumerate(imp_for_task):
            if mol_id >= n_mols:
                break
            smiles = global_smiles[mol_id]
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if mol is None:
                continue
            n_heavy = mol.GetNumAtoms()  # implicit-H mol -> all atoms are heavy
            centers = [a for a in impdict.get("tight", []) if 0 <= a < n_heavy]
            if not centers:
                continue
            grp_matches = _group_matches(mol)  # functional groups to keep intact (computed once/mol)
            # collect the SET of substructures this molecule contributes (dedup within molecule)
            subs = set()
            for a in centers:
                for r in radii:
                    s = _circular_env_smiles(mol, a, r, grp_matches)
                    if s:
                        subs.add(s)
            if not subs:
                continue
            pred = int(preds_for_task[mol_id]) if mol_id < len(preds_for_task) else 0
            for s in subs:
                frag_counts_per_task[task_idx][s] += 1
                if pred == 1:
                    frag_pos_counts_per_task[task_idx][s] += 1
                frag_examples[s].add(mol_id)

    all_frags = set()
    for c in frag_counts_per_task:
        all_frags.update(c.keys())

    df_rows = []
    for frag in all_frags:
        counts = [frag_counts_per_task[t][frag] for t in range(n_tasks)]
        pos_counts = [frag_pos_counts_per_task[t][frag] for t in range(n_tasks)]
        frag_alert_matches = compare_fragment_to_alerts(frag, alerts_compiled, alert_fps)
        matched_alerts = [m[0] for m in frag_alert_matches] if frag_alert_matches else []
        df_rows.append({
            "fragment": frag,
            "total_count": sum(counts),
            "total_pos_count": sum(pos_counts),
            **{f"count_t{t}": counts[t] for t in range(n_tasks)},
            **{f"pos_t{t}": pos_counts[t] for t in range(n_tasks)},
            "matched_alerts": ";".join(sorted(set(matched_alerts))),
            "examples": ";".join(str(x) for x in sorted(list(frag_examples.get(frag, set())))[:10]),
        })

    per_task_top_sets = [set(f for f, _ in frag_counts_per_task[t].most_common(top_k))
                         for t in range(n_tasks)]
    return df_rows, per_task_top_sets, frag_examples

def get_fragment_smiles(mol, atom_indices):
    """Extract a clean, canonical, hydrogen-free SMILES for the selected atoms.

    Uses RDKit's MolFragmentToSmiles so bond orders and aromaticity are inherited from the parent
    molecule (no more invalid 'cc(...)'-style fragments), then round-trips + RemoveHs so the result
    is a valid canonical SMILES that deduplicates and substructure-matches against the alert SMARTS.
    """
    if mol is None or not atom_indices:
        return None
    n_atoms = mol.GetNumAtoms()
    atom_indices = [a for a in set(atom_indices) if 0 <= a < n_atoms]
    if not atom_indices:
        return None
    try:
        frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_indices, canonical=True)
        fm = Chem.MolFromSmiles(frag_smi, sanitize=False)
        if fm is None:
            return None
        Chem.SanitizeMol(fm, catchErrors=True)
        fm = Chem.RemoveHs(fm)
        smi = Chem.MolToSmiles(fm)
        return smi or None
    except Exception as e:
        print(f"[get_fragment_smiles] Extraction failed: {e}")
        return None

from rdkit.Chem import rdMolDescriptors, DataStructs

def compare_fragment_to_alerts(frag_smiles, alerts_compiled,
                               alert_fps=None,
                               fp_radius=2,
                               fp_bits=2048,
                               similarity_threshold=0.65):
    """
    Minimal-change improvement:
    - keeps existing substructure matching
    - adds fingerprint similarity fallback (no new SMARTS needed)
    - returns SAME FORMAT as before
    """

    if frag_smiles is None:
        return []

    try:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return []
    except:
        return []

    results = []
    frag_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(frag, fp_radius, nBits=fp_bits)

    for alert_name, patt in alerts_compiled:

        # 1) try your original substructure matching
        try:
            if frag.HasSubstructMatch(patt):
                results.append((alert_name, "alert_in_fragment", 1.0))
                continue
            if patt.HasSubstructMatch(frag):
                results.append((alert_name, "fragment_in_alert", 1.0))
                continue
        except:
            pass

        # 2) fingerprint similarity check (fallback)
        if alert_fps is not None:
            sim = DataStructs.TanimotoSimilarity(frag_fp, alert_fps[alert_name])
            if sim >= similarity_threshold:
                results.append((alert_name, "similar_to_alert", sim))

    return results


# Organic atom set (everything else => metal / inorganic artifact): H,B,C,N,O,F,P,S,Cl,Br,I
_ORGANIC_ATOMS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def _fragment_mol_props(frag_smiles, alerts_compiled):
    """Canonical mol + properties for a fragment SMILES, or None if invalid.
    matched_alerts is recomputed robustly (both substructure directions) on the canonical fragment."""
    if not isinstance(frag_smiles, str) or not frag_smiles:
        return None
    m = Chem.MolFromSmiles(frag_smiles, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, catchErrors=True)
        m = Chem.RemoveHs(m)
    except Exception:
        return None
    zs = [a.GetAtomicNum() for a in m.GetAtoms()]
    matched = [n for n, p in alerts_compiled
               if p is not None and (m.HasSubstructMatch(p) or p.HasSubstructMatch(m))]
    return {
        "mol": m,
        "n_heavy": m.GetNumHeavyAtoms(),
        "n_hetero": sum(1 for z in zs if z not in (1, 6)),
        "has_ring": m.GetRingInfo().NumRings() > 0,
        "has_metal": any(z not in _ORGANIC_ATOMS for z in zs),
        "net_charge": Chem.GetFormalCharge(m),
        "matched_alerts": ";".join(sorted(set(matched))),
    }


def select_novel_fragments(df_rows, alerts_compiled, min_heavy=5, min_hetero=2, min_support=5):
    """Single, unified definition of a 'novel' fragment, used by BOTH the CSV and the figure (and the
    standalone refilter script). Canonicalizes + deduplicates fragments (summing counts), then keeps a
    fragment iff it matches no known alert, is fully organic, has >= min_heavy heavy atoms and (a ring
    OR >= min_hetero heteroatoms), has >= min_support total occurrences, and is enriched in positive
    predictions (pos_frac >= dataset base rate). Returns a DataFrame (canonical `fragment`, `mol`,
    properties, `pos_frac`) ranked by enrichment.
    """
    df = pd.DataFrame(df_rows)
    if df.empty:
        return df
    per_task = [c for c in df.columns if c.startswith("count_t") or c.startswith("pos_t")]

    # Canonicalize + deduplicate by canonical SMILES, summing counts.
    merged = {}
    for _, r in df.iterrows():
        info = _fragment_mol_props(r.get("fragment"), alerts_compiled)
        if info is None:
            continue
        csmi = Chem.MolToSmiles(info["mol"])
        e = merged.get(csmi)
        if e is None:
            e = {"fragment": csmi, "mol": info["mol"], "n_heavy": info["n_heavy"],
                 "n_hetero": info["n_hetero"], "has_ring": info["has_ring"],
                 "has_metal": info["has_metal"], "net_charge": info["net_charge"],
                 "matched_alerts": info["matched_alerts"],
                 "total_count": 0, "total_pos_count": 0, **{c: 0 for c in per_task}}
            merged[csmi] = e
        e["total_count"] += int(r.get("total_count", 0) or 0)
        e["total_pos_count"] += int(r.get("total_pos_count", 0) or 0)
        for c in per_task:
            e[c] += int(r.get(c, 0) or 0)

    cand = pd.DataFrame(list(merged.values()))
    if cand.empty:
        return cand
    base_rate = cand["total_pos_count"].sum() / max(1, cand["total_count"].sum())
    cand["pos_frac"] = cand["total_pos_count"] / cand["total_count"].clip(lower=1)
    keep = (
        (cand["matched_alerts"] == "")
        & (~cand["has_metal"])
        & (cand["net_charge"] == 0)
        & (cand["n_heavy"] >= min_heavy)
        & (cand["has_ring"] | (cand["n_hetero"] >= min_hetero))
        & (cand["total_count"] >= min_support)
        & (cand["pos_frac"] >= base_rate)
    )
    return cand[keep].sort_values(["pos_frac", "total_pos_count"], ascending=False).reset_index(drop=True)


def save_fragment_artifacts(df_rows, per_task_sets, frag_examples, output_dir, alerts_compiled, global_smiles, topN_grid=24):
    # Save CSV summary
    df_frags = pd.DataFrame(df_rows)
    frag_csv = os.path.join(output_dir, "explainer_discovered_fragments_summary.csv")
    df_frags.sort_values("total_pos_count", ascending=False).to_csv(frag_csv, index=False)

    # Identify novel candidates with the unified, stricter definition (organic, non-trivial,
    # enriched in positives, not a known alert). Same definition is used for the figure below.
    df_sorted = df_frags.sort_values("total_pos_count", ascending=False)
    novel = select_novel_fragments(df_rows, alerts_compiled)
    novel_cols = ["fragment", "n_heavy", "n_hetero", "has_ring",
                  "total_count", "total_pos_count", "pos_frac", "matched_alerts"]
    novel[[c for c in novel_cols if c in novel.columns]].to_csv(
        os.path.join(output_dir, "explainer_novel_fragment_candidates.csv"), index=False)

    # Save grid image of top fragments (top by total_pos_count)
    top_frags = df_sorted.head(topN_grid)["fragment"].tolist()
    # mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in top_frags]

    mols = []
    for s in top_frags:
        if s is None:
            return []
        try:
            frag_base = Chem.MolFromSmiles(s)
            if frag_base is None:
                # Try sanitization: sometimes fragments miss valence info
                try:
                    frag_base = Chem.MolFromSmiles(s, sanitize=False)
                    Chem.SanitizeMol(frag_base, catchErrors=True)
                except Exception:
                    print(f"[Warning] Could not sanitize invalid fragment: {s}")
                    return []
            # frag_mol = Chem.AddHs(frag_base)
        except Exception as e:
            print(f"[Error] Could not parse fragment SMILES {s}: {e}")
            return []

        frag_mol = frag_base

        if frag_mol is None:
            return []

        mols.append(frag_mol)

    #mols = [Chem.MolFromSmiles(s) for s in top_frags]
    legends = []
    for _, r in df_sorted.head(topN_grid).iterrows():
        legends.append(f"pos:{r['total_pos_count']},tot:{r['total_count']}")
    img = Draw.MolsToGridImage(mols, molsPerRow=min(6, len(mols)), subImgSize=(200, 200), legends=legends)
    img.save(os.path.join(output_dir, "top_discovered_fragments_grid.pdf"))

# Split fragments into known-alert vs novel, using the SAME unified novel definition as the CSV.
def get_fragment_info_lists(df_rows, alerts_compiled, global_smiles, min_heavy_atoms=2):
    # Novel: identical definition/ranking as explainer_novel_fragment_candidates.csv.
    novel_df = select_novel_fragments(df_rows, alerts_compiled)
    novel_frags = [{
        "fragment": r["fragment"], "mol": r["mol"], "alerts": "",
        "total_pos_count": r["total_pos_count"], "total_count": r["total_count"],
        "pos_frac": r.get("pos_frac", 0.0),
    } for _, r in novel_df.iterrows()]

    # Known-alert fragments: organic, >= min_heavy_atoms, and matching a known alert (recomputed).
    alert_frags = []
    for r in df_rows:
        info = _fragment_mol_props(r.get("fragment"), alerts_compiled)
        if info is None or info["has_metal"] or info["n_heavy"] < min_heavy_atoms:
            continue
        if info["matched_alerts"]:
            alert_frags.append({
                "fragment": r["fragment"], "mol": info["mol"], "alerts": info["matched_alerts"],
                "total_pos_count": r["total_pos_count"], "total_count": r["total_count"],
            })
    alert_frags.sort(key=lambda x: x["total_pos_count"], reverse=True)
    return alert_frags, novel_frags

def plot_combined_known_vs_novel(alert_frags, novel_frags, output_dir, top_n_each=30):
    out_path = os.path.join(output_dir, "fragments_known_vs_novel_combined.png")
    n_show_alert = min(len(alert_frags), top_n_each)
    n_show_novel = min(len(novel_frags), top_n_each)
    show_list = alert_frags[:n_show_alert] + novel_frags[:n_show_novel]

    mols = [e["mol"] for e in show_list]
    legends = []
    for e in show_list:
        smi = e["fragment"]
        if e["alerts"]:
            #alerts_str = ", ".join(e["alerts"])
            #legends.append(f"{smi}\n(alerts: {alerts_str})")
            legends.append(f"alerts: {e["alerts"]}")
        else:
            #alerts_str = ", novel".join(e["total_pos_count"])
            #legends.append(f"{smi}\n(novel) {alerts_str}")
            #legends.append(f"{smi}\n(novel)")
            #legends.append(f"{e[n_pos_count]})
            legends.append(f"novel, total pos count: {e["total_pos_count"]}")

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

# Plot summary
def assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, output_dir):

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
    rows_correct_toxic_scaled = []
    rows_correct_nontoxic_scaled = []
    rows_incorrect_scaled = []
    smiles_correct_toxic = []
    smiles_correct_nontoxic = []
    smiles_incorrect = []
    avg_attr_correct_toxic = []
    avg_attr_correct_nontoxic = []
    avg_attr_incorrect = []
    # Per-strain + average cell render data, captured so single-molecule SVGs
    # can be regenerated from the CSV without rerunning the analysis.
    cells_correct_toxic = []
    cells_correct_nontoxic = []
    cells_incorrect = []
    strain_names = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']

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
        strain_cells_scaled = []
        strain_cell_data = []
        for task in range(n_tasks):
            pdf = per_task_dfs[task]
            pdf_task = pdf[task]
            mol_df = pdf_task[pdf_task['mol_id'] == mol_id]
            if mol_df.empty:
                strain_cells.append(blank_image(cell_size))
                strain_cells_scaled.append(blank_image(cell_size))
                strain_cell_data.append({"strain": strain_names[task], "blank": True})
                continue
            per_task_labels_t = per_task_labels[task]
            correct_label = int(per_task_labels_t[task][mol_id])
            per_task_preds_t = per_task_preds[task]
            pred = int(per_task_preds_t[task][mol_id])
            if correct_label == -1:
                im = blank_image(cell_size)
                strain_cells.append(im)
                strain_cells_scaled.append(blank_image(cell_size))
                strain_cell_data.append({"strain": strain_names[task], "blank": True})
                continue
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

                # Add important atoms for this strain (uniform red)
                imp = per_task_impatoms[task]
                imp_t = imp[task]
                tight = imp_t[mol_id]['tight'] if mol_id < len(imp_t) else []
                loose = imp_t[mol_id]['loose'] if mol_id < len(imp_t) else []
                for a in tight:
                    atom_colors[a] = (1.0, 0.3, 0.3)  # red
                    highlight_atoms.append(a)
                #for a in loose:
                #    if a not in atom_colors:
                #        atom_colors[a] = (1.0, 0.8, 0.3)  # orange
                #    highlight_atoms.append(a)

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

                # --- Scaled cell: importance-scaled red ---
                _tight_scores = imp_t[mol_id].get('tight_scores', {}) if mol_id < len(imp_t) else {}
                highlight_atoms_sc = list(highlight_atoms)
                atom_colors_sc = dict(atom_colors)
                highlight_bonds_sc = list(highlight_bonds)
                bond_colors_sc = dict(bond_colors)
                if tight and _tight_scores:
                    _scores = [_tight_scores.get(a, 0.0) for a in tight]
                    _s_min, _s_max = min(_scores), max(_scores)
                    _s_range = _s_max - _s_min if _s_max != _s_min else 1.0
                    for a, sc in zip(tight, _scores):
                        t_norm = (sc - _s_min) / _s_range   # 0=least, 1=most important
                        atom_colors_sc[a] = (1.0 - 0.1 * t_norm, 0.8 * (1.0 - t_norm), 0.8 * (1.0 - t_norm))
                        if a not in highlight_atoms_sc:
                            highlight_atoms_sc.append(a)
                else:
                    for a in tight:
                        atom_colors_sc[a] = (1.0, 0.3, 0.3)
                        if a not in highlight_atoms_sc:
                            highlight_atoms_sc.append(a)
                im_sc = draw_with_colors(mol, highlight_atoms_sc, atom_colors_sc,
                                         highlight_bonds_sc, bond_colors_sc, size=cell_size)
                _draw_sc = ImageDraw.Draw(im_sc)
                try:
                    _font_sc = ImageFont.truetype('DejaVuSans.ttf', 14)
                except Exception:
                    _font_sc = ImageFont.load_default()
                _draw_sc.rectangle([(0, 0), (im_sc.size[0], 18)], fill=(255, 255, 255))
                _draw_sc.text((4, 0), text, fill=(0, 0, 0), font=_font_sc)
                strain_cells_scaled.append(im_sc)

                # Capture the exact scaled-cell inputs so the per-strain SVG matches the PDF cell.
                strain_cell_data.append({
                    "strain": strain_names[task],
                    "atoms": sorted({int(a) for a in highlight_atoms_sc}),
                    "atom_colors": {int(a): [float(c) for c in col] for a, col in atom_colors_sc.items()},
                    "bonds": sorted({int(b) for b in highlight_bonds_sc}),
                    "bond_colors": {int(b): [float(c) for c in col] for b, col in bond_colors_sc.items()},
                    "header": f"{strain_names[task]}  {text}",
                })

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

        # --- Averaged attribution cell (scaled PDF only) ---
        # Per-node score = mean of tight_scores across all 5 strains (0 if absent in a strain)
        _node_avg_scores = {}
        for _task_avg in range(n_tasks):
            _imp_t_avg = per_task_impatoms[_task_avg][_task_avg]
            if mol_id < len(_imp_t_avg):
                for _node, _sc in _imp_t_avg[mol_id].get('tight_scores', {}).items():
                    _node_avg_scores[_node] = _node_avg_scores.get(_node, 0.0) + _sc / n_tasks

        _avg_ha, _avg_ac, _avg_hb, _avg_bc = [], {}, [], {}
        # Carry alert colours from task 0 for context
        _mol_df_0 = per_task_dfs[0][0]
        _mol_df_0 = _mol_df_0[_mol_df_0['mol_id'] == mol_id]
        for _, _row0 in _mol_df_0.iterrows():
            if _row0['alert_present']:
                _name0 = _row0['alert']
                _patt0 = next((p for n, p in alerts_compiled if n == _name0), None)
                if _patt0:
                    for _match0 in mol.GetSubstructMatches(_patt0):
                        _col0 = alert_colors.get(_name0, (0.5, 0.5, 0.5))
                        for _a0 in _match0:
                            _avg_ha.append(_a0)
                            _avg_ac[_a0] = _col0
                        for _b0 in mol.GetBonds():
                            _u0, _v0 = _b0.GetBeginAtomIdx(), _b0.GetEndAtomIdx()
                            if _u0 in _match0 and _v0 in _match0:
                                _bid0 = _b0.GetIdx()
                                _avg_hb.append(_bid0)
                                _avg_bc[_bid0] = _col0

        if _node_avg_scores:
            _avg_s_min = min(_node_avg_scores.values())
            _avg_s_max = max(_node_avg_scores.values())
            _avg_s_range = _avg_s_max - _avg_s_min if _avg_s_max != _avg_s_min else 1.0
            for _node, _sc in _node_avg_scores.items():
                _t_n = (_sc - _avg_s_min) / _avg_s_range
                _avg_ac[_node] = (1.0 - 0.1 * _t_n, 0.8 * (1.0 - _t_n), 0.8 * (1.0 - _t_n))
                if _node not in _avg_ha:
                    _avg_ha.append(_node)

        avg_cell = draw_with_colors(mol, _avg_ha, _avg_ac, _avg_hb, _avg_bc, size=cell_size)
        _draw_avg = ImageDraw.Draw(avg_cell)
        try:
            _font_avg = ImageFont.truetype('DejaVuSans.ttf', 14)
        except Exception:
            _font_avg = ImageFont.load_default()
        _draw_avg.rectangle([(0, 0), (avg_cell.size[0], 18)], fill=(255, 255, 255))
        _draw_avg.text((4, 0), "Avg (all strains)", fill=(0, 0, 0), font=_font_avg)

        # Capture the average cell inputs (matches the scaled-PDF avg cell exactly).
        avg_cell_data = {
            "atoms": sorted({int(a) for a in _avg_ha}),
            "atom_colors": {int(a): [float(c) for c in col] for a, col in _avg_ac.items()},
            "bonds": sorted({int(b) for b in _avg_hb}),
            "bond_colors": {int(b): [float(c) for c in col] for b, col in _avg_bc.items()},
            "header": "Avg (all strains)",
            "consensus": int(consensus),
            "overall_label": int(overall_label),
        }
        cell_record = {"strains": strain_cell_data, "avg": avg_cell_data}

        row_imgs = strain_cells + [cons_im, im_alerts_present]
        row_concat = hstack_images(row_imgs, pad=4)
        row_concat_scaled = hstack_images(strain_cells_scaled + [cons_im, im_alerts_present, avg_cell], pad=4)

        if consensus == 1 and overall_label == 1:
            rows_correct_toxic.append(row_concat)
            rows_correct_toxic_scaled.append(row_concat_scaled)
            smiles_correct_toxic.append(global_smiles[mol_id])
            avg_attr_correct_toxic.append(_node_avg_scores)
            cells_correct_toxic.append(cell_record)
        elif consensus == 0 and overall_label == 0:
            rows_correct_nontoxic.append(row_concat)
            rows_correct_nontoxic_scaled.append(row_concat_scaled)
            smiles_correct_nontoxic.append(global_smiles[mol_id])
            avg_attr_correct_nontoxic.append(_node_avg_scores)
            cells_correct_nontoxic.append(cell_record)
        else:
            rows_incorrect.append(row_concat)
            rows_incorrect_scaled.append(row_concat_scaled)
            smiles_incorrect.append(global_smiles[mol_id])
            avg_attr_incorrect.append(_node_avg_scores)
            cells_incorrect.append(cell_record)

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

    # Save importance-scaled PDFs and SMILES CSVs
    outdir_scaled = os.path.join(output_dir, "summary_rows_scaled")
    os.makedirs(outdir_scaled, exist_ok=True)
    for rows, smiles_list, avg_attr_list, cells_list, stem in [
        (rows_correct_toxic_scaled,    smiles_correct_toxic,    avg_attr_correct_toxic,    cells_correct_toxic,    "summary_correct_toxic"),
        (rows_correct_nontoxic_scaled, smiles_correct_nontoxic, avg_attr_correct_nontoxic, cells_correct_nontoxic, "summary_correct_nontoxic"),
        (rows_incorrect_scaled,        smiles_incorrect,        avg_attr_incorrect,        cells_incorrect,        "summary_incorrect"),
    ]:
        save_rows_to_pdf(rows, os.path.join(outdir_scaled, f"{stem}.pdf"), alert_colors)
        if smiles_list:
            pd.DataFrame({
                "SMILES": smiles_list,
                "avg_attributions": [json.dumps({str(k): v for k, v in d.items()}) for d in avg_attr_list],
                "row_cells": [json.dumps(c) for c in cells_list],
            }).to_csv(os.path.join(outdir_scaled, f"{stem}_smiles.csv"), index=False)

    return alerts_present_by_mol

# Save to PDF
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

# Make legend on PDF
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

# Highlight alerts
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

def save_attribution_color_scale_svg(plot_dir):
    """Standalone vertical color-scale legend matching the per-atom red shading used in the
    alert_averaged_plots_positional molecule plots: color(f) = (1, 1-f**2, 1-f**2), f in [0,1]."""
    from matplotlib.colors import LinearSegmentedColormap

    n = 256
    f = np.linspace(0.0, 1.0, n)
    colors = [(1.0, 1.0 - v ** 2, 1.0 - v ** 2) for v in f]   # white -> dark red, squared like atoms
    cmap = LinearSegmentedColormap.from_list("attribution_red", colors, N=n)

    fig, ax = plt.subplots(figsize=(1.6, 6))
    gradient = f.reshape(-1, 1)                                # value axis linear in freq
    ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=[0, 1, 0.0, 1.0])
    ax.set_xticks([])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks(np.linspace(0.0, 1.0, 6))                   # 0.0, 0.2, ... 1.0
    ax.set_ylabel("Relative attribution", rotation=270, labelpad=18, va="bottom")
    plt.tight_layout()

    svg_path = os.path.join(plot_dir, "attribution_color_scale.svg")
    fig.savefig(svg_path, format="svg", transparent=True)
    plt.close(fig)


def analyze_per_atom_overlap_by_alert(per_task_impatoms, alerts_compiled, global_smiles, per_task_dfs, per_task_labels, output_dir):
    """
    Compute per-SMARTS-position GNN importance frequency across all toxic molecules matching
    each structural alert.

    Averaging key: SMARTS tuple index i, so position 0 always maps to the same chemical atom
    in the query pattern across every molecule — no fragile hashing needed.

    Ring atoms in the same ring as a SMARTS-matched atom but not explicitly in the SMARTS
    are tracked with key (nearest_smarts_pos, ring_distance, element), which is consistent
    across molecules of the same ring size and allows the full ring environment to be shown.

    Representative molecule: smallest matching molecule, so the heatmap is drawn on the
    simplest possible context.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "alert_averaged_plots_positional")
    os.makedirs(plot_dir, exist_ok=True)
    save_attribution_color_scale_svg(plot_dir)   # one legend for the whole folder

    alert_dict = defaultdict(list)
    for name, patt in alerts_compiled:
        alert_dict[name].append(patt)

    ALIPHATIC_ALERTS = {
        "Aliphatic azo and azoxy groups",
        "Aliphatic halogens",
        "Aliphatic N-nitro groups",
        "Alpha, beta unsaturated aliphatic alkoxy groups",
    }

    all_records = []

    for alert_name, patt_list in alert_dict.items():
        n_patts = len(patt_list)

        # pos_imp_sum[patt_idx][smarts_pos] = total times that SMARTS position was GNN-important
        # pos_imp_cnt[patt_idx][smarts_pos] = total times that SMARTS position was seen
        pos_imp_sum = [defaultdict(int) for _ in range(n_patts)]
        pos_imp_cnt = [defaultdict(int) for _ in range(n_patts)]

        # ring_imp_sum/cnt[patt_idx][(nearest_smarts_pos, ring_distance, element)]
        ring_imp_sum = [defaultdict(int) for _ in range(n_patts)]
        ring_imp_cnt = [defaultdict(int) for _ in range(n_patts)]

        # mol_id -> (mol, best_match, n_atoms); up to 3 smallest per pattern
        rep_mol_by_patt = [{} for _ in range(n_patts)]

        # --- Step 1: Aggregate per molecule / strain ---
        for task_dict in per_task_impatoms:
            task_id = list(task_dict.keys())[0]
            imp_list = task_dict[task_id]

            for mol_id, smi in enumerate(global_smiles):
                if mol_id >= len(imp_list):
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                per_task_labels_t = per_task_labels[task_id]
                if int(per_task_labels_t[task_id][mol_id]) == -1:
                    continue

                df_task = per_task_dfs[task_id][task_id]
                overall_rows = df_task[df_task['mol_id'] == mol_id]
                if overall_rows.empty:
                    continue
                if int(overall_rows.iloc[0]['label_overall']) == 0:
                    continue

                max_valid_idx = mol.GetNumAtoms() - 1
                tight_set = {idx for idx in imp_list[mol_id].get("tight", [])
                             if 0 <= idx <= max_valid_idx}

                for patt_idx, patt in enumerate(patt_list):
                    matches = mol.GetSubstructMatches(patt)
                    if not matches:
                        continue

                    valid_matches = [
                        m for m in matches
                        if alert_name not in ALIPHATIC_ALERTS
                        or is_aliphatic_context_valid(mol, m, alert_name)
                    ]
                    if not valid_matches:
                        continue

                    # Best match = most GNN-important atoms in the raw SMARTS match
                    best_match = max(valid_matches, key=lambda m: len(set(m) & tight_set))
                    matched_set = set(best_match)

                    # SMARTS-position importance
                    for smarts_pos, mol_atom_idx in enumerate(best_match):
                        pos_imp_sum[patt_idx][smarts_pos] += (1 if mol_atom_idx in tight_set else 0)
                        pos_imp_cnt[patt_idx][smarts_pos] += 1

                    # Ring-extension importance
                    match_pos_map = {atom_idx: pos for pos, atom_idx in enumerate(best_match)}
                    for ring in mol.GetRingInfo().AtomRings():
                        ring_set = set(ring)
                        ring_match = ring_set & matched_set
                        ring_extra = ring_set - matched_set
                        if not ring_match or not ring_extra:
                            continue
                        ring_list = list(ring)
                        ring_n = len(ring_list)
                        for extra_atom in ring_extra:
                            nm_pos = ring_list.index(extra_atom)
                            best_sp, best_dist = None, ring_n
                            for ma in ring_match:
                                m_pos = ring_list.index(ma)
                                d = min((nm_pos - m_pos) % ring_n, (m_pos - nm_pos) % ring_n)
                                if d < best_dist:
                                    best_dist, best_sp = d, match_pos_map[ma]
                            if best_sp is None:
                                continue
                            elem = mol.GetAtomWithIdx(extra_atom).GetSymbol()
                            rkey = (best_sp, best_dist, elem)
                            ring_imp_sum[patt_idx][rkey] += (1 if extra_atom in tight_set else 0)
                            ring_imp_cnt[patt_idx][rkey] += 1

                    # Keep the 3 smallest unique molecules as representatives
                    n_atoms = mol.GetNumAtoms()
                    rep_dict = rep_mol_by_patt[patt_idx]
                    if mol_id not in rep_dict:
                        rep_dict[mol_id] = (mol, best_match, n_atoms)
                        if len(rep_dict) > 3:
                            worst_id = max(rep_dict, key=lambda k: rep_dict[k][2])
                            del rep_dict[worst_id]

        # --- Step 2: Visualize and record ---
        for patt_idx, patt in enumerate(patt_list):
            rep_entries = sorted(rep_mol_by_patt[patt_idx].values(), key=lambda x: x[2])
            if not rep_entries:
                continue

            patt_suffix = f"_patt{patt_idx}" if len(patt_list) > 1 else ""

            # Write CSV records once (importance values are per-SMARTS-position, same for all reps)
            first_mol, first_match, _ = rep_entries[0]
            first_matched_set = set(first_match)
            first_match_pos_map = {atom_idx: pos for pos, atom_idx in enumerate(first_match)}
            for smarts_pos, mol_atom_idx in enumerate(first_match):
                cnt = pos_imp_cnt[patt_idx].get(smarts_pos, 0)
                if cnt == 0:
                    continue
                freq = pos_imp_sum[patt_idx][smarts_pos] / cnt
                all_records.append({
                    "alert": alert_name, "patt_idx": patt_idx,
                    "key_type": "smarts_pos", "key": smarts_pos,
                    "atom_idx_rep_mol": mol_atom_idx,
                    "importance_freq": round(freq, 4),
                    "n_instances": cnt,
                })
            for ring in first_mol.GetRingInfo().AtomRings():
                ring_set = set(ring)
                ring_match = ring_set & first_matched_set
                ring_extra = ring_set - first_matched_set
                if not ring_match or not ring_extra:
                    continue
                ring_list = list(ring)
                ring_n = len(ring_list)
                for extra_atom in ring_extra:
                    nm_pos = ring_list.index(extra_atom)
                    best_sp, best_dist = None, ring_n
                    for ma in ring_match:
                        m_pos = ring_list.index(ma)
                        d = min((nm_pos - m_pos) % ring_n, (m_pos - nm_pos) % ring_n)
                        if d < best_dist:
                            best_dist, best_sp = d, first_match_pos_map[ma]
                    if best_sp is None:
                        continue
                    elem = first_mol.GetAtomWithIdx(extra_atom).GetSymbol()
                    rkey = (best_sp, best_dist, elem)
                    cnt = ring_imp_cnt[patt_idx].get(rkey, 0)
                    if cnt == 0:
                        continue
                    freq = ring_imp_sum[patt_idx][rkey] / cnt
                    all_records.append({
                        "alert": alert_name, "patt_idx": patt_idx,
                        "key_type": "ring_ext", "key": str(rkey),
                        "atom_idx_rep_mol": extra_atom,
                        "importance_freq": round(freq, 4),
                        "n_instances": cnt,
                    })

            # Draw the averaged heatmap on each of the (up to 3) representative molecules
            for rep_idx, (r_mol, r_match, _) in enumerate(rep_entries):
                r_matched_set = set(r_match)
                r_match_pos_map = {atom_idx: pos for pos, atom_idx in enumerate(r_match)}

                r_atom_colors = {}
                r_highlight_atoms = []

                for smarts_pos, mol_atom_idx in enumerate(r_match):
                    cnt = pos_imp_cnt[patt_idx].get(smarts_pos, 0)
                    if cnt == 0:
                        continue
                    freq = pos_imp_sum[patt_idx][smarts_pos] / cnt
                    val = freq ** 2
                    if val > 0.0025:
                        r_atom_colors[mol_atom_idx] = (1.0, 1.0 - val, 1.0 - val)
                        r_highlight_atoms.append(mol_atom_idx)

                for ring in r_mol.GetRingInfo().AtomRings():
                    ring_set = set(ring)
                    ring_match = ring_set & r_matched_set
                    ring_extra = ring_set - r_matched_set
                    if not ring_match or not ring_extra:
                        continue
                    ring_list = list(ring)
                    ring_n = len(ring_list)
                    for extra_atom in ring_extra:
                        nm_pos = ring_list.index(extra_atom)
                        best_sp, best_dist = None, ring_n
                        for ma in ring_match:
                            m_pos = ring_list.index(ma)
                            d = min((nm_pos - m_pos) % ring_n, (m_pos - nm_pos) % ring_n)
                            if d < best_dist:
                                best_dist, best_sp = d, r_match_pos_map[ma]
                        if best_sp is None:
                            continue
                        elem = r_mol.GetAtomWithIdx(extra_atom).GetSymbol()
                        rkey = (best_sp, best_dist, elem)
                        cnt = ring_imp_cnt[patt_idx].get(rkey, 0)
                        if cnt == 0:
                            continue
                        freq = ring_imp_sum[patt_idx][rkey] / cnt
                        val = freq ** 2
                        if val > 0.0025:
                            r_atom_colors[extra_atom] = (1.0, 1.0 - val, 1.0 - val)
                            r_highlight_atoms.append(extra_atom)

                try:
                    drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
                except AttributeError:
                    drawer = rdMolDraw2D.MolDraw2D(600, 600)

                rdMolDraw2D.PrepareAndDrawMolecule(
                    drawer, r_mol,
                    highlightAtoms=r_highlight_atoms,
                    highlightAtomColors=r_atom_colors,
                    highlightAtomRadii={i: 0.4 for i in r_highlight_atoms},
                )
                drawer.FinishDrawing()
                outpath = os.path.join(
                    plot_dir,
                    f"{alert_name.replace('/', '_')}{patt_suffix}_smarts_pos_avg_rep{rep_idx}.png",
                )
                with open(outpath, "wb") as fh:
                    fh.write(drawer.GetDrawingText())

                drawer_svg = rdMolDraw2D.MolDraw2DSVG(600, 600)
                rdMolDraw2D.PrepareAndDrawMolecule(
                    drawer_svg, r_mol,
                    highlightAtoms=r_highlight_atoms,
                    highlightAtomColors=r_atom_colors,
                    highlightAtomRadii={i: 0.4 for i in r_highlight_atoms},
                )
                drawer_svg.FinishDrawing()
                svg_path = os.path.join(
                    plot_dir,
                    f"{alert_name.replace('/', '_')}{patt_suffix}_smarts_pos_avg_rep{rep_idx}.svg",
                )
                with open(svg_path, "w") as fh:
                    fh.write(drawer_svg.GetDrawingText())

            plot_important_atoms_by_alert(
                alert_name, [patt],
                global_smiles, per_task_impatoms, per_task_labels, output_dir,
            )

    csv_path = os.path.join(output_dir, "alert_atom_smarts_pos_avg_summary.csv")
    pd.DataFrame.from_records(all_records).to_csv(csv_path, index=False)

# --- NEW FUNCTION: plot_important_atoms_by_alert ---
def plot_important_atoms_by_alert(alert_name, patt_list, global_smiles, per_task_impatoms, per_task_labels, output_dir, mols_per_row=6,
                                  max_mols=48):
    mols_to_plot = []

    alert_plot_dir = os.path.join(output_dir, "alert_instance_grids")
    os.makedirs(alert_plot_dir, exist_ok=True)

    # Collect tight important atoms and predictions by mol_id and task_id
    mol_task_data = defaultdict(lambda: defaultdict(dict))

    for task_idx in range(len(per_task_impatoms)):
        imp_list = per_task_impatoms[task_idx][task_idx]

        # NOTE: If predictions are needed, they would need to be passed here as well.
        # Since they are not, we only plot if the tight set is non-empty.

        for mol_id, imp_entry in enumerate(imp_list):
            mol_task_data[mol_id][task_idx]['tight'] = imp_entry.get("tight", [])

    # Iterate through all molecules to find matches
    for mol_id, smi in enumerate(global_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue

        is_matched = False
        for patt in patt_list:
            if mol.HasSubstructMatch(patt):
                is_matched = True
                break

        if is_matched:
            # Check all tasks for this molecule
            for task_idx in range(len(per_task_impatoms)):
                imp_atoms_set = set(mol_task_data[mol_id][task_idx]['tight'])

                # Only plot if the GNN found atoms important in this instance
                if imp_atoms_set:
                    if len(mols_to_plot) >= max_mols:
                        break

                    per_task_labels_t = per_task_labels[task_idx]
                    correct_label = int(per_task_labels_t[task_idx][mol_id])
                    if correct_label == -1:
                        continue

                    # Find the alert match (raw SMARTS atoms) with most overlap with GNN atoms
                    best_match_atoms = set()
                    for patt in patt_list:
                        for match in mol.GetSubstructMatches(patt):
                            match_set = set(match)
                            if len(match_set & imp_atoms_set) > len(best_match_atoms & imp_atoms_set):
                                best_match_atoms = match_set

                    # Three-way coloring:
                    #   orange = overlap (in both alert and GNN)
                    #   blue   = alert-only (in alert but not GNN)
                    #   red    = GNN-only (in GNN but not alert)
                    overlap_atoms = imp_atoms_set & best_match_atoms
                    alert_only = best_match_atoms - imp_atoms_set
                    gnn_only = imp_atoms_set - best_match_atoms

                    atom_colors = {}
                    for a in alert_only:
                        atom_colors[a] = (0.4, 0.6, 1.0)   # blue
                    for a in gnn_only:
                        atom_colors[a] = (1.0, 0.3, 0.3)   # red
                    for a in overlap_atoms:
                        atom_colors[a] = (1.0, 0.65, 0.0)  # orange

                    mol_copy = Chem.Mol(mol)
                    mols_to_plot.append({
                        'mol': mol_copy,
                        'highlight_atoms': list(best_match_atoms | imp_atoms_set),
                        'atom_colors': atom_colors,
                        'legend': f"Mol {mol_id} T{task_idx + 1} | orange=overlap blue=alert red=GNN",
                    })
            if len(mols_to_plot) >= max_mols: break

    if not mols_to_plot:
        print(f"  No molecules found for alert '{alert_name}'. Skipping grid plot.")
        return

    # Draw the grid image
    grid_images = []
    cell_size = (300, 300)
    for m in mols_to_plot:
        im = draw_with_colors(
            m['mol'],
            m['highlight_atoms'],
            m['atom_colors'],
            highlight_bonds=[],
            highlight_bond_colors={},
            size=cell_size
        )
        # Add legend text to the top of the image
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 14)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([(0, 0), (im.size[0], 18)], fill=(255, 255, 255))
        draw.text((4, 0), m['legend'], fill=(0, 0, 0), font=font)

        grid_images.append(im)

    # Stitch images into a grid
    num_mols = len(grid_images)
    mols_per_row = min(mols_per_row, num_mols)
    num_rows = (num_mols + mols_per_row - 1) // mols_per_row

    grid_w = mols_per_row * cell_size[0]
    grid_h = num_rows * cell_size[1]

    grid_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(grid_images):
        row = idx // mols_per_row
        col = idx % mols_per_row
        x = col * cell_size[0]
        y = row * cell_size[1]
        grid_img.paste(img, (x, y))

    outpath = os.path.join(alert_plot_dir, f"{alert_name.replace('/', '_')}_instance_grid.png")
    grid_img.save(outpath)
    print(f"  Grid plot saved for alert: '{alert_name}' to {outpath}")

    # If many molecules, also save as PDF for better viewing
    if num_mols > mols_per_row * 3:
        pdf_path = os.path.join(alert_plot_dir, f"{alert_name.replace('/', '_')}_instance_grid.pdf")
        # Simple save: one image per page
        grid_img.save(pdf_path, save_all=True, append_images=[grid_img.convert("RGB")])
        print(f"  PDF plot saved for alert: '{alert_name}' to {pdf_path}")


def is_aliphatic_context_valid(mol, match, alert_name):
    """
    Checks if a match for an ALIPHATIC alert is structurally valid (i.e., not attached
    to or part of an aromatic ring, and acyclic if required by the alert type).
    """

    # 1. Aromaticity Check (Match Atoms + Neighbors)
    atoms_to_check = set(match)
    # Check neighbors for aromaticity (Benzylic/Azo attachments)
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1:  # Heavy atoms only
                atoms_to_check.add(neighbor.GetIdx())

    # Perform aromatic check on the expanded environment
    for atom_idx in atoms_to_check:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetIsAromatic():
            return False  # Reject if any matched atom or neighbor is aromatic

    # 2. Alkoxy Ring Check (Specific Alert)
    if alert_name == "Alpha, beta unsaturated aliphatic alkoxy groups":
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Check if the atom is in ANY ring (aliphatic or aromatic)
            if atom.IsInRing():
                return False  # Reject if any matched atom is in any ring

    return True  # Passed all aliphatic/acyclic constraints

def compute_overall_alert_performance(alerts_compiled, alerts_present_by_mol, per_task_dfs, n_tasks, global_smiles, output_dir):
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

    EXCLUDED_MOLECULES = {
        "Aliphatic azo and azoxy groups": {85},
        "Alpha, beta unsaturated aliphatic alkoxy groups": {239, 821, 850}
    }

    for mol_id, (present_alerts, _, label_list) in enumerate(alerts_present_by_mol):

        # Determine which alerts are present in molecule
        alerts_present_by_mol = []
        smiles = global_smiles[mol_id]
        mol = Chem.MolFromSmiles(smiles)
        for name, patt in alerts_compiled:
            matches = mol.GetSubstructMatches(patt)
            if matches:
                stats[name]["n_present"] += 5

        mol_overlaps = defaultdict(list)
        # Loop through all tasks/strains
        for task in range(n_tasks):
            df_task = per_task_dfs[task][task]
            mol_df = df_task[df_task["mol_id"] == mol_id]
            for _, row in mol_df.iterrows():
                a = row["alert"]
                if not row["alert_present"]: # Only consider alerts with overlap > 0
                    continue

                skip_ids = EXCLUDED_MOLECULES.get(a, set())
                if mol_id in skip_ids:
                    continue

                overlap = row.get("tight_score", 0.0)
                mol_overlaps[a].append(overlap)
                stats[a]["n_detected"] += 1

                if row["label_overall"] > 0:
                    is_toxic = True
                else:
                    is_toxic = False

        # Aggregate per alert
        for alert in present_alerts:
            if alert not in stats:
                continue
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
        mean_overlap_tox = (np.sum(v["overlaps_toxic"]))/5 if v["overlaps_toxic"] else 0
        mean_overlap_non = (np.sum(v["overlaps_nontoxic"]))/5 if v["overlaps_nontoxic"] else 0
        rows.append({
            "n_present": n_pres,
            "n_detected": n_det,
            "alert": alert,
            "percent_correct": pct_correct,
            "mean_overlap": mean_overlap,
            "mean_overlap_toxic": mean_overlap_tox,
            "mean_overlap_nontoxic": mean_overlap_non
        })

    df_perf = pd.DataFrame(rows).set_index("alert")

    if output_dir:
        outpath = os.path.join(output_dir, "alert_strain_summary.csv")
        df_perf.to_csv(outpath, index=False)
    else:
        print(df_perf.head())

    return df_perf

def compute_detection_frequencies(alerts_compiled, per_task_dfs, n_tasks):
    all_alerts = [name for name, _ in alerts_compiled]
    detection_freqs = pd.DataFrame(0.0, index=all_alerts, columns=[f"Strain {i+1}" for i in range(n_tasks)])

    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()
        df_task["alert_detected"] = df_task["alert_present"] > 0

        for alert in all_alerts:
            df_alert = df_task[df_task["alert"] == alert]
            #if len(df_alert) == 0:
            #    continue
            mol_detected = df_alert.groupby("mol_id")["alert_detected"].max()
            detection_freqs.loc[alert, f"Strain {task+1}"] = mol_detected.mean()  # average across all mols

    return detection_freqs

def _order_alerts(index, order):
    """Return `index` reordered to follow `order`; alerts not in `order` are appended
    at the end in their original relative order."""
    if order is None:
        return list(index)
    order = list(order)
    order_set = set(order)
    index_set = set(index)
    in_order = [a for a in order if a in index_set]
    leftovers = [a for a in index if a not in order_set]
    return in_order + leftovers


def plot_alert_performance_bars(df_perf, output_dir=None):
    order = df_perf.sort_values(ascending=False).index
    df_perf = df_perf.loc[order]

    #fig, ax = plt.subplots(figsize=(12, 0.4 * len(df_perf)))  # longer and skinnier
    fig, ax = plt.subplots(figsize=(5, 15))

    y = np.arange(len(df_perf))
    barh_kwargs = dict(height=0.25, edgecolor="black")

    #ax.barh(y - 0.25, df_perf["percent_correct"], color="#6baed6", label="% Correctly Identified", **barh_kwargs)
    ax.barh(y, df_perf * 100, color="#fd8d3c", label="Mean Overlap (Toxic)", **barh_kwargs)
    #ax.barh(y + 0.25, df_perf["mean_overlap_nontoxic"] * 100, color="#969696", label="Mean Overlap (Non-Toxic)", **barh_kwargs)

    ax.set_yticks(y)
    ax.set_yticklabels(df_perf.index)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage / Overlap ×100")
    ax.set_title("Structural Alert Detection Performance (Overlap > 0 Definition)")
    #ax.legend()
    plt.tight_layout()

    if output_dir:
        outpath = os.path.join(output_dir, "alert_performance_bars.pdf")
        plt.savefig(outpath, dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

    return order

def compute_toxic_overlap_by_strain(alerts_compiled, per_task_dfs, n_tasks, alerts_present_by_mol):
    all_alerts = [name for name, _ in alerts_compiled]
    overlap_scores = pd.DataFrame(np.nan, index=all_alerts, columns=[f"Strain {i+1}" for i in range(n_tasks)])

    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()

        for alert in all_alerts:
            df_alert = df_task[(df_task["alert"] == alert) & (df_task["alert_matched"]) & (df_task["label_overall"] == 1)]
            if len(df_alert) == 0:
                continue

            overlap_scores.loc[alert, f"Strain {task + 1}"] = df_alert["tight_score"].mean()

    mean_overlap_scores = overlap_scores.mean(axis=1)

    return overlap_scores, mean_overlap_scores


def export_overlap_diagnostic_xlsx(alerts_compiled, per_task_dfs, n_tasks, output_path,
                                    global_smiles, per_task_impatoms):
    """
    Export per-molecule overlap diagnostics for direct comparison with manual spreadsheet.
    Records n_raw_match_atoms and n_expanded_atoms separately so the denominator used by
    the code can be compared against whatever atom count was used in the spreadsheet.
    """
    records = []

    for alert_name, smarts in alerts_compiled:
        for task in range(n_tasks):
            df_task = per_task_dfs[task][task]
            df_alert = df_task[(df_task["alert"] == alert_name) & (df_task["alert_matched"])]
            imp_list = per_task_impatoms[task][task]

            for _, row in df_alert.iterrows():
                mid = int(row["mol_id"])
                mol = Chem.MolFromSmiles(global_smiles[mid]) if mid < len(global_smiles) else None

                n_raw = None
                n_expanded = None
                n_gnn_overlap = None

                if mol is not None:
                    matches = mol.GetSubstructMatches(smarts)
                    if matches:
                        imp_dict = imp_list[mid] if mid < len(imp_list) else {"tight": []}
                        tight = set(imp_dict.get("tight", []))
                        best_match = max(
                            matches,
                            key=lambda m: len(set(m) & tight)
                        )
                        raw_set = set(best_match)
                        expanded_set = expand_match_atoms(mol, best_match)
                        n_raw = len(raw_set)
                        n_expanded = len(expanded_set)
                        n_gnn_overlap = len(raw_set & tight)

                records.append({
                    "alert": alert_name,
                    "mol_id": mid,
                    "strain": task + 1,
                    "label_overall": int(row["label_overall"]),
                    "n_raw_match_atoms": n_raw,
                    "n_expanded_atoms": n_expanded,
                    "n_gnn_overlap_atoms": n_gnn_overlap,
                    "tight_score_code": float(row["tight_score"]),
                    "tight_score_check": (n_gnn_overlap / n_raw) if n_raw else None,
                })

    df_diag = pd.DataFrame(records)
    df_diag.to_excel(output_path, index=False)
    print(f"\nDiagnostic overlap file saved to {output_path}")
    if df_diag.empty:
        print("(no alert-matched molecules found; per-alert summary skipped)")
        return
    print("\nMean atom counts per alert (raw SMARTS match vs. after expansion):")
    print(df_diag.groupby("alert")[["n_raw_match_atoms", "n_expanded_atoms"]].mean().round(2).to_string())


def plot_toxic_overlap_heatmap(overlap_scores, mean_overlap_scores, output_dir=None):
    order = mean_overlap_scores.sort_values(ascending=False).index
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
        outpath = os.path.join(output_dir, "toxic_overlap_by_strain_heatmap.pdf")
        plt.savefig(outpath, dpi=600, transparent=True)  # high-res + transparent bg
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# AUC by alert/strain and 4-category analysis
# ---------------------------------------------------------------------------

def compute_auc_by_alert_strain(alerts_compiled, per_task_dfs, n_tasks):
    """AUROC for each (alert, strain) among alert-matched molecules with valid per-strain labels."""
    from sklearn.metrics import roc_auc_score
    all_alerts = [name for name, _ in alerts_compiled]
    auc_df = pd.DataFrame(np.nan, index=all_alerts,
                          columns=[f"Strain {i + 1}" for i in range(n_tasks)])
    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()
        df_task = df_task[df_task["label"].isin([0, 1])]
        for alert in all_alerts:
            sub = df_task[(df_task["alert"] == alert) & df_task["alert_matched"]]
            if len(sub) < 10 or sub["label"].nunique() < 2:
                continue
            try:
                auc_df.loc[alert, f"Strain {task + 1}"] = roc_auc_score(sub["label"], sub["prob"])
            except Exception:
                pass
    return auc_df


def plot_auc_heatmap_by_strain(auc_df, output_dir=None, order=None):
    """Heatmap of AUROC per alert × strain (mirrors the overlap heatmap)."""
    mean_auc = auc_df.mean(axis=1, skipna=True)
    shown = mean_auc[mean_auc.notna()].index          # which alerts to display
    ordered = _order_alerts(shown, order)             # match the alert performance bars order
    auc_plot = auc_df.loc[ordered]

    plt.figure(figsize=(8, max(6, len(ordered) * 0.4)))
    sns.heatmap(
        auc_plot,
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        cbar_kws={"label": "AUROC"},
        linewidths=0.5,
        linecolor="lightgray",
        annot=True,
        fmt=".2f",
    )
    plt.title("AUROC per Structural Alert and Strain\n(alert-matched molecules only)")
    plt.xlabel("Strain")
    plt.ylabel("Structural Alert")
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "auc_by_alert_strain_heatmap.pdf"),
                    dpi=600, transparent=True)
        plt.close()
    else:
        plt.show()


def compute_alert_category_auc(alerts_compiled, per_task_dfs, n_tasks):
    """
    For each alert, categorise all molecules into 4 groups based on alert presence
    and overall Ames outcome, then compute a one-vs-rest AUROC per category using
    the model's average probability across all strain heads.

    Categories
    ----------
    A = alert_matched & label_overall==1  (alert present, mutagenic)
    B = alert_matched & label_overall==0  (alert present, non-mutagenic)
    C = ~alert_matched & label_overall==1 (alert absent, mutagenic)
    D = ~alert_matched & label_overall==0 (alert absent, non-mutagenic)

    AUC_X = AUROC(is_X ~ avg_model_prob) across all molecules for this alert.
    Interpretation:
      High AUC_A  → model correctly assigns high probability to alert+mutagenic.
      Low  AUC_B  → model is NOT fooled by alert-positive non-mutagenic molecules.
      High AUC_C  → model finds mutagenic molecules even without structural alerts.
      Low  AUC_D  → model correctly assigns low probability to non-alert, non-mutagenic.
    """
    from sklearn.metrics import roc_auc_score

    all_alerts = [name for name, _ in alerts_compiled]

    # Build a molecule-level DataFrame with avg prob across tasks
    task_frames = []
    for task in range(n_tasks):
        df_t = per_task_dfs[task][task][
            ["mol_id", "alert", "prob", "label_overall", "alert_matched"]
        ].copy().rename(columns={"prob": f"prob_{task}"})
        task_frames.append(df_t)

    df_merged = task_frames[0].copy()
    for t in range(1, n_tasks):
        df_merged = df_merged.merge(
            task_frames[t][["mol_id", "alert", f"prob_{t}"]],
            on=["mol_id", "alert"], how="left",
        )
    prob_cols = [f"prob_{t}" for t in range(n_tasks)]
    df_merged["avg_prob"] = df_merged[prob_cols].mean(axis=1)
    df_merged = df_merged[df_merged["label_overall"].isin([0, 1])]

    rows = []
    for alert in all_alerts:
        df_a = df_merged[df_merged["alert"] == alert].copy()
        if len(df_a) == 0:
            continue
        total = len(df_a)
        is_A = (df_a["alert_matched"]) & (df_a["label_overall"] == 1)
        is_B = (df_a["alert_matched"]) & (df_a["label_overall"] == 0)
        is_C = (~df_a["alert_matched"]) & (df_a["label_overall"] == 1)
        is_D = (~df_a["alert_matched"]) & (df_a["label_overall"] == 0)

        row = {
            "alert": alert,
            "n_A": int(is_A.sum()), "frac_A": is_A.sum() / total,
            "n_B": int(is_B.sum()), "frac_B": is_B.sum() / total,
            "n_C": int(is_C.sum()), "frac_C": is_C.sum() / total,
            "n_D": int(is_D.sum()), "frac_D": is_D.sum() / total,
        }
        probs_arr = df_a["avg_prob"].values
        for cat, mask in [("A", is_A), ("B", is_B), ("C", is_C), ("D", is_D)]:
            n_pos, n_neg = mask.sum(), (~mask).sum()
            if n_pos >= 5 and n_neg >= 5:
                try:
                    row[f"AUC_{cat}"] = roc_auc_score(mask.astype(int), probs_arr)
                except Exception:
                    row[f"AUC_{cat}"] = np.nan
            else:
                row[f"AUC_{cat}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_alert_category_heatmap(cat_df, output_dir=None, order=None):
    """
    Two side-by-side heatmaps per alert:
      Left  — fraction of molecules in each category (A, B, C, D)
      Right — one-vs-rest AUROC for each category
    Alerts ordered to match the alert performance bars (`order`).
    """
    cat_df = cat_df.dropna(subset=["AUC_A"], how="all").copy().set_index("alert")
    cat_df = cat_df.loc[_order_alerts(cat_df.index, order)]

    frac_cols = ["frac_A", "frac_B", "frac_C", "frac_D"]
    auc_cols  = ["AUC_A",  "AUC_B",  "AUC_C",  "AUC_D"]
    n = len(cat_df)
    fig_h = max(6, n * 0.35)

    _, axes = plt.subplots(1, 2, figsize=(12, fig_h))

    frac_data = cat_df[frac_cols].rename(columns={
        "frac_A": "A\n(alert+mut)", "frac_B": "B\n(alert+non)",
        "frac_C": "C\n(no alert+mut)", "frac_D": "D\n(no alert+non)",
    })
    sns.heatmap(frac_data * 100, ax=axes[0], cmap="Blues", vmin=0, vmax=100,
                annot=True, fmt=".1f", linewidths=0.4, linecolor="lightgray",
                cbar_kws={"label": "% of molecules"})
    axes[0].set_title("Fraction of molecules per category (%)")
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("")

    auc_data = cat_df[auc_cols].rename(columns={
        "AUC_A": "A", "AUC_B": "B", "AUC_C": "C", "AUC_D": "D",
    })
    sns.heatmap(auc_data, ax=axes[1], cmap="RdYlGn", vmin=0, vmax=1,
                annot=True, fmt=".2f", linewidths=0.4, linecolor="lightgray",
                cbar_kws={"label": "One-vs-rest AUROC"})
    axes[1].set_title("One-vs-rest AUROC per category")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("")

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "alert_category_auc_heatmap.pdf"),
                    dpi=600, transparent=True)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Helper functions for input feature importance analysis (Integrated Gradients)
# ---------------------------------------------------------------------------

def combine(smiles_list, important_atoms_per_mol, predictions, probs, correct_val, correct_val_overall):
    rows = []

    for i, (smiles, imp_dict, pred, prob, label, label_overall) in enumerate(
            zip(smiles_list, important_atoms_per_mol, predictions, probs, correct_val, correct_val_overall)):
        mol = Chem.MolFromSmiles(smiles)

        rows.append({
            "mol_id": i,
            "smiles": smiles,
            "imp_dict": imp_dict,
            "prediction": pred, "prob": prob,
            "label": label,
            "label_overall": label_overall,
            })

    return pd.DataFrame(rows)

def plot_task_bars(importances_dict, feature_names, title_prefix, filename_prefix, plot_dir):

    for task_id, values in importances_dict.items():
        if not isinstance(values, np.ndarray):
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), feature_names, rotation=60, ha="right")
        plt.ylabel("Importance Score")
        plt.title(f"{title_prefix} — Task {task_id}")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{filename_prefix}_task_{task_id}.png"), dpi=300)
        plt.close()

def plot_overall_bars(values, feature_names, title, filename, plot_dir):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), feature_names, rotation=60, ha="right")
    plt.ylabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

def plot_heatmap(importances_dict, feature_names, title, filename, plot_dir):
    # Collect all tasks in sorted order
    task_ids = sorted(importances_dict.keys())

    # Build matrix (num_tasks × num_features)
    matrix = []
    for t in task_ids:
        vals = importances_dict[t]
        if isinstance(vals, np.ndarray):
            matrix.append(vals)
        else:
            matrix.append(np.zeros(len(feature_names)))

    matrix = np.array(matrix)  # shape: (T, F)
    num_tasks, num_features = matrix.shape

    # -------- FIX LABEL MISMATCHES -------- #
    if len(feature_names) != num_features:
        print(
            f"[WARN] feature_names ({len(feature_names)}) does not match matrix width ({num_features}). "
            "Auto-adjusting."
        )
        if len(feature_names) > num_features:
            feature_names = feature_names[:num_features]
        else:
            # pad missing names
            feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), num_features)]

    # same for tasks
    if len(task_ids) != num_tasks:
        print(
            f"[WARN] task_ids ({len(task_ids)}) does not match matrix height ({num_tasks}). "
            "Auto-adjusting."
        )
        task_ids = task_ids[:num_tasks]

    # -------- PLOT -------- #
    plt.figure(figsize=(12, 6))

    sns.heatmap(
        matrix,
        annot=False,
        cmap="viridis",
        xticklabels=feature_names,
        yticklabels=[f"Task {t}" for t in task_ids],
    )

    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Tasks")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


def plot_shap_violin(node_matrix, node_feature_names,
                     edge_matrix, edge_feature_names,
                     node_feat_values, edge_feat_values,
                     node_groups, title, filename, plot_dir):
    """
    SHAP layered violin plot.
    x = signed IG attribution value; color = actual input feature value (blue=low, red=high).
    KDE violin outline shows density; dots are beeswarm-jittered inside.
    One-hot node feature groups are averaged into a single score.
    """
    from scipy.stats import gaussian_kde

    # --- Group one-hot node features (attributions and feature values together) ---
    grouped_node_names = []
    grouped_node_cols = []
    grouped_node_feat_cols = []
    grouped_indices = set()

    for group_name, group_feats in node_groups:
        idxs = [i for i, n in enumerate(node_feature_names) if n in set(group_feats)]
        if idxs:
            grouped_node_cols.append(node_matrix[:, idxs].mean(axis=1))
            # Color value for a one-hot family: composition-weighted mean category index (columns are in
            # ordinal order). The plain mean of indicators is a constant 1/group_size and carries no info.
            _v = node_feat_values[:, idxs]
            _w = _v.sum(axis=1)
            _w = np.where(_w == 0, 1.0, _w)
            grouped_node_feat_cols.append((_v @ np.arange(len(idxs))) / _w)
            grouped_node_names.append(group_name)
            grouped_indices.update(idxs)

    for i, name in enumerate(node_feature_names):
        if i not in grouped_indices:
            grouped_node_cols.append(node_matrix[:, i])
            grouped_node_feat_cols.append(node_feat_values[:, i])
            grouped_node_names.append(name)

    node_processed = np.column_stack(grouped_node_cols)       # (n_mols, n_grouped_node_feats)
    node_feat_processed = np.column_stack(grouped_node_feat_cols)

    edge_names = list(edge_feature_names)
    feat_type_map = {n: "Node" for n in grouped_node_names}
    feat_type_map.update({n: "Edge" for n in edge_names})

    # Combined attribution and feature-value matrices: (n_mols, n_all_feats)
    all_feat_names = grouped_node_names + edge_names
    all_shap = np.hstack([node_processed, edge_matrix])
    all_feat_vals = np.hstack([node_feat_processed, edge_feat_values])

    # Sort features by mean absolute attribution descending (most important at top)
    mean_abs = np.abs(all_shap).mean(axis=0)
    sort_order = np.argsort(-mean_abs)
    sorted_names = [all_feat_names[i] for i in sort_order]
    sorted_shap = all_shap[:, sort_order]
    sorted_feat_vals = all_feat_vals[:, sort_order]

    # Normalise feature values per feature for colormap (0=low, 1=high). Robust 5-95 percentile clip
    # (matches shap.summary_plot); min-max is dominated by the heavy right-skew of molecular feature
    # values and washes the bulk to one color.
    feat_lo = np.nanpercentile(sorted_feat_vals, 5, axis=0, keepdims=True)
    feat_hi = np.nanpercentile(sorted_feat_vals, 95, axis=0, keepdims=True)
    feat_range = np.where(feat_hi - feat_lo == 0, 1.0, feat_hi - feat_lo)
    norm_feat_vals = np.clip((sorted_feat_vals - feat_lo) / feat_range, 0.0, 1.0)  # (n_mols, n_feats)

    n_feats = len(sorted_names)
    fig_height = max(6, n_feats * 0.5)
    _, ax = plt.subplots(figsize=(10, fig_height))

    cmap = plt.cm.coolwarm
    rng = np.random.default_rng(seed=42)

    for feat_idx in range(n_feats):
        vals = sorted_shap[:, feat_idx]
        fv_norm = norm_feat_vals[:, feat_idx]
        y_center = feat_idx

        # KDE violin outline (gray fill)
        if vals.std() > 1e-10 and len(vals) > 2:
            try:
                kde = gaussian_kde(vals, bw_method='scott')
                x_range = np.linspace(vals.min(), vals.max(), 200)
                kde_density = kde(x_range)
                kde_density = kde_density / kde_density.max()
                ax.fill_between(x_range,
                                y_center - 0.4 * kde_density,
                                y_center + 0.4 * kde_density,
                                color="lightgray", alpha=0.6, zorder=1)
                # Beeswarm jitter proportional to local density
                kde_at_pts = kde(vals)
                kde_at_pts = kde_at_pts / kde_at_pts.max()
                jitter = rng.uniform(-1, 1, size=len(vals)) * 0.38 * kde_at_pts
            except Exception:
                jitter = rng.uniform(-0.35, 0.35, size=len(vals))
        else:
            jitter = rng.uniform(-0.35, 0.35, size=len(vals))

        marker = "o" if feat_type_map[sorted_names[feat_idx]] == "Node" else "D"
        ax.scatter(vals, y_center + jitter,
                   c=fv_norm, cmap=cmap, vmin=0, vmax=1,
                   alpha=0.7, s=10, linewidths=0, marker=marker, zorder=3)

    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.axvline(0, color="darkgray", linewidth=0.8, zorder=2)

    # Colorbar for input feature values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Feature value (normalized)", shrink=0.4, pad=0.01)

    # Legend: node vs edge marker
    legend_handles = [
        Line2D([], [], color="gray", marker="o", linestyle="None",
               markersize=6, label="Node feature"),
        Line2D([], [], color="gray", marker="D", linestyle="None",
               markersize=6, label="Edge feature"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    ax.set_xlabel("IG Attribution")
    ax.set_ylabel("")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.savefig(os.path.join(plot_dir, os.path.splitext(filename)[0] + ".svg"))
    plt.close()

    # --- Export CSV ---
    records = []
    for feat_idx in range(n_feats):
        feat_name = sorted_names[feat_idx]
        feat_type = feat_type_map[feat_name]
        for sv, fvv in zip(sorted_shap[:, feat_idx], sorted_feat_vals[:, feat_idx]):
            records.append({"Feature": feat_name, "Type": feat_type,
                            "SHAP Value": sv, "Feature Value": fvv})
    csv_path = os.path.join(plot_dir, filename.replace(".png", "_values.csv"))
    pd.DataFrame(records).to_csv(csv_path, index=False)


def main():
    ### Build/load model
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file = args.input_file

    with open(input_file, 'r') as input_stream:
        input_data = yaml.load(input_stream, Loader=yaml.Loader)

    # Resolve data_file: CLI flag overrides YAML
    data_file = args.data_file if args.data_file else input_data.get("data_file")

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
    dihedral_angle_features = database_data.get("DihedralAngleFeatures", True)
    n_dist_feats = database_data.get("nDistanceFeatures", 1)  # 1 raw or N for RBF
    n_edge_features = n_dist_feats
    if bond_angle_features: n_edge_features += 1
    if dihedral_angle_features: n_edge_features += 1

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
    input_mode = input_data.get("inputMode", None)
    if input_mode is None:
        # Backward compatibility with old YAML configs
        input_mode = "descriptor" if input_data.get("useMolecularDescriptors", False) else "gnn"

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
                            activation, input_mode, n_inputs)

    checkpoint = torch.load(args.checkpoint_file, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if not args.analyze_input_features_only:
        model = model.to(device)
    
        per_task_dfs = []
        per_task_impatoms = []
        per_task_preds = []
        per_task_labels = []
        global_smiles = []
    
        ### GNNExplainer analysis
        for task_id in range(5):
    
            task = task_id
            model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers,
                          n_shared_layers, n_target_specific_layers, input_mode)
    
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
            probs = []
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
    
                    _prob = task_output.item()
                    prediction = int(_prob > 0.5)  # 1 = toxic, 0 = non-toxic
                    predictions.append(prediction)
                    probs.append(_prob)
    
                edge_mask = explanation.edge_mask.detach().cpu().numpy()
    
                # Tight filter
                k_edges_tight = int(0.15 * edge_mask.size)  # max(8, int(0.15 * edge_mask.size))  # ~10–15%
                top_e_tight = np.argsort(-edge_mask)[:k_edges_tight]
    
                imp_edges_tight = data.edge_index[:, torch.tensor(top_e_tight, device=data.edge_index.device)]
                imp_nodes_tight = sorted(set(imp_edges_tight.view(-1).tolist()))
    
                G = to_networkx(data, to_undirected=True)
                #sub_tight = G.subgraph(imp_nodes_tight).copy()
                # if sub_tight.number_of_nodes() > 0:
                #    lcc_tight = max(nx.connected_components(sub_tight), key=len)
                #    important_atoms_tight = sorted(list(lcc_tight))
                # else:
                #    important_atoms_tight = []
                # if sub_tight.number_of_nodes() > 0:
                #    important_atoms_tight = imp_nodes_tight
                # else:
                #    important_atoms_tight = []
                #if sub_tight.number_of_nodes() > 0:
                    # Keep *all* connected components, not just the largest
                    #comps = max(nx.connected_components(sub_tight), key=len)
                    #important_atoms_tight = sorted(list(comps))
                important_atoms_tight = imp_nodes_tight
                    # comps = list(nx.connected_components(sub_tight))
                    # important_atoms_tight = sorted(set().union(*comps))
                #else:
                    #important_atoms_tight = []

                # Per-node importance: sum of edge_mask for all incident edges
                _node_edge_sum = {}
                for _e in range(edge_mask.size):
                    _u = int(data.edge_index[0, _e])
                    _v = int(data.edge_index[1, _e])
                    _node_edge_sum[_u] = _node_edge_sum.get(_u, 0.0) + float(edge_mask[_e])
                    _node_edge_sum[_v] = _node_edge_sum.get(_v, 0.0) + float(edge_mask[_e])
                tight_scores = {n: _node_edge_sum.get(n, 0.0) for n in important_atoms_tight}

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
                    "loose": important_atoms_loose,
                    "tight_scores": tight_scores,
                })
    
                # Extract SMILES
                # CSV file with structure data
                csv_file = data_file
                df = pd.read_csv(csv_file)
                filepath = os.path.basename(data.file_name)
    
                molecule_index = molecule_index = int(
                    re.search(r'(\d+)_', filepath).group(1))  # get molecule number from input file name
    
                # Resolve column names to support both CSV formats
                _strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
                smiles_col = 'SMILES RDKit' if 'SMILES RDKit' in df.columns else 'SMILES'
                row = df.iloc[molecule_index - 1]
    
                # Extract the SMILES string from the specific row and column
                smiles_string = row[smiles_col]
                smiles_list.append(smiles_string)
    
                correct = row[_strain_cols[task]]
                correct_val.append(correct)
    
                correct_overall = row['Overall']
                correct_val_overall.append(correct_overall)
    
            per_task_impatoms.append({task_id: important_atoms_per_mol})
            per_task_preds.append({task_id: predictions})
            per_task_labels.append({task_id: correct_val})
            global_smiles = smiles_list
    
            alerts = load_alerts()
    
            df = evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, probs, correct_val, correct_val_overall)  # Compute overlap scores by comparing alerts and important nodes, store all in df
            df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
            per_task_dfs.append({task_id: df})
    
            # Consider an alert present if tight_score > 0 or loose_score > 0
            #df['alert_present'] = (df['tight_score'] > 0) | (df['loose_score'] > 0)
            df['alert_present'] = (df['tight_score'] > 0)
    
        # Known structural alerts
        alerts_compiled = load_alerts()

        ### Fragment analysis (NOVEL-FRAGMENT DETECTION ONLY)
        # Uses the EXTENDED alert list + recurring-substructure mining. This block is isolated: every
        # other analysis below keeps using the base `alerts_compiled` / `load_alerts()`.
        extended_alerts = load_extended_alerts()
        extended_alert_fps = compute_alert_fps(extended_alerts)
        # Recurring circular substructures mined around the model's tight important atoms.
        df_rows, per_task_top_sets, frag_examples = build_substructure_catalog(
            per_task_impatoms, per_task_preds, per_task_labels, global_smiles,
            extended_alerts, extended_alert_fps, top_k=200)

        # Save and plot fragment analysis (initial)
        save_fragment_artifacts(df_rows, per_task_top_sets, frag_examples, args.output_dir, extended_alerts, global_smiles, topN_grid=24)

        # Divide into novel vs not, eliminate alerts with < 2 heavy atoms
        alert_frags, novel_frags = get_fragment_info_lists(df_rows, extended_alerts, global_smiles, min_heavy_atoms=4)
    
        # Save and plot novel vs not fragments
        n_alert, n_novel = plot_combined_known_vs_novel(alert_frags, novel_frags, args.output_dir, top_n_each=30)
    
        ### Overlap with known structural alerts per molecule per task
        # Save PDF summary for all molecules with highlighted alerts
        alerts_present_by_mol = assemble_and_save_summary(per_task_dfs, per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, args.output_dir)
    
        # Plot per-atom overlap on known structural alerts
        analyze_per_atom_overlap_by_alert(per_task_impatoms, alerts_compiled, global_smiles, per_task_dfs, per_task_labels, args.output_dir)
    
        ### Strain-specific structural alert detection analysis
        # For each alert, calculate % correctly identified and mean overlap score (toxic vs nontoxic)
        #df_perf = compute_overall_alert_performance(alerts_compiled, alerts_present_by_mol, per_task_dfs, 5, global_smiles, args.output_dir)
    
        # Save and plot per-alert bar graph, output order of alerts for second plot (sorted by mean toxic overlap)
        #order = plot_alert_performance_bars(df_perf, args.output_dir)
    
        overlap_scores, mean_overlap_scores = compute_toxic_overlap_by_strain(alerts_compiled, per_task_dfs, 5, alerts_present_by_mol)
    
        export_overlap_diagnostic_xlsx(
            alerts_compiled, per_task_dfs, 5,
            os.path.join(args.output_dir, "overlap_diagnostic.xlsx"),
            global_smiles, per_task_impatoms
        )
    
        # Filter to alerts with nonzero overall overlap before plotting
        nonzero_alerts = mean_overlap_scores[mean_overlap_scores > 0].index
        overlap_scores_nz = overlap_scores.loc[nonzero_alerts]
        mean_overlap_scores_nz = mean_overlap_scores.loc[nonzero_alerts]
    
        # Save and plot heatmap of % overlap for each strain
        plot_toxic_overlap_heatmap(overlap_scores_nz, mean_overlap_scores_nz, args.output_dir)
    
        # Reference ordering for all alert figures (mean toxic overlap, descending)
        alert_order = plot_alert_performance_bars(mean_overlap_scores_nz, args.output_dir)

        # AUC heatmap (AUROC per alert × strain, alert-matched molecules)
        auc_by_strain = compute_auc_by_alert_strain(alerts_compiled, per_task_dfs, 5)
        nonzero_auc_alerts = auc_by_strain.dropna(how="all").index
        if len(nonzero_auc_alerts) > 0:
            plot_auc_heatmap_by_strain(auc_by_strain.loc[nonzero_auc_alerts], args.output_dir,
                                       order=alert_order)
            auc_by_strain.to_csv(
                os.path.join(args.output_dir, "auc_by_alert_strain.csv"), index=True
            )

        # 4-category analysis (A/B/C/D fractions + one-vs-rest AUROC)
        cat_df = compute_alert_category_auc(alerts_compiled, per_task_dfs, 5)
        if len(cat_df) > 0:
            cat_df.to_csv(
                os.path.join(args.output_dir, "alert_category_auc.csv"), index=False
            )
            # Summary row: mean across all alerts
            auc_cols = ["AUC_A", "AUC_B", "AUC_C", "AUC_D"]
            summary = cat_df[auc_cols].agg(["mean", "std"]).T.rename(
                columns={"mean": "Mean", "std": "Std"}
            ).reset_index().rename(columns={"index": "Category"})
            summary.to_csv(
                os.path.join(args.output_dir, "alert_category_auc_summary.csv"), index=False
            )
            plot_alert_category_heatmap(cat_df, args.output_dir, order=alert_order)

    # ---------------------------------------------------------------------------
    # Optional: Input Feature Importance via Integrated Gradients
    # ---------------------------------------------------------------------------
    if args.analyze_input_features or args.analyze_input_features_only:

        def integrated_gradients(task_model, data, baseline_x=None, baseline_edge=None,
                                 steps=50, device=torch.device("cpu")):
            """
            Compute Integrated Gradients for node features (data.x) and edge features (data.edge_attr)
            Returns:
              node_attr (num_nodes, num_node_features) : attributions (can be positive/negative)
              edge_attr (num_edges, num_edge_features) : attributions
            """
            # Prepare inputs
            x = data.x.detach().to(device)
            edge_attr = data.edge_attr.detach().to(device)
            edge_index = data.edge_index.to(device)
            batch = data.batch.to(device) if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long,
                                                                                     device=device)

            # -------------------------------
            # 2. Construct dihedral mask
            # -------------------------------
            _dihedral_idx = n_edge_features - 1  # dihedral is always the last edge feature
            dihedral_mask = ~torch.isnan(edge_attr[:, _dihedral_idx])  # True where valid

            # -------------------------------
            # 3. Create a clean version (NaNs -> 0)
            # -------------------------------
            clean_edge_attr = edge_attr.clone()
            clean_edge_attr[:, _dihedral_idx] = torch.nan_to_num(clean_edge_attr[:, _dihedral_idx], nan=0.0)

            # Baselines (zeros if not provided)
            if baseline_x is None:
                baseline_x = torch.zeros_like(x, device=device)
            else:
                baseline_x = baseline_x.to(device)

            if baseline_edge is None:
                baseline_edge = torch.zeros_like(edge_attr, device=device)
            else:
                baseline_edge = baseline_edge.to(device)

            # Steps and scaled inputs
            alphas = torch.linspace(0.0, 1.0, steps, device=device)

            # Accumulate gradients
            total_grad_x = torch.zeros_like(x, device=device)
            total_grad_edge = torch.zeros_like(edge_attr, device=device)

            # Ensure model in eval
            was_training = task_model.training
            task_model.eval()

            # -------------------------------
            # 6. Integrated gradients loop
            # -------------------------------
            for alpha in alphas:
                x_interp = baseline_x + alpha * (x - baseline_x)
                edge_interp = baseline_edge + alpha * (clean_edge_attr - baseline_edge)

                x_interp = x_interp.clone().detach().requires_grad_(True)
                edge_interp = edge_interp.clone().detach().requires_grad_(True)

                out = task_model(
                    x=x_interp,
                    edge_index=edge_index,
                    edge_attr=edge_interp,
                    batch=batch
                )

                out = out.squeeze()
                if out.numel() > 1:
                    out = out[0]

                task_model.zero_grad()

                grad_x_alpha, grad_edge_alpha = torch.autograd.grad(
                    outputs=out,
                    inputs=(x_interp, edge_interp),
                    grad_outputs=torch.ones_like(out),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )

                if grad_x_alpha is None:
                    grad_x_alpha = torch.zeros_like(x, device=device)
                if grad_edge_alpha is None:
                    grad_edge_alpha = torch.zeros_like(clean_edge_attr, device=device)

                total_grad_x += grad_x_alpha
                total_grad_edge += grad_edge_alpha

            # -------------------------------
            # 7. IG = average gradient × (input - baseline)
            # -------------------------------
            avg_grad_x = total_grad_x / steps
            avg_grad_edge = total_grad_edge / steps

            node_attributions = (x - baseline_x) * avg_grad_x
            edge_attributions = (clean_edge_attr - baseline_edge) * avg_grad_edge

            # -------------------------------
            # 8. Mask dihedral importance
            #    Set attribution = 0 for edges where dihedral doesn't exist.
            #    NOTE: the dihedral *input* construction bug (only the last edge per molecule got a
            #    value) was fixed in XG_graphs.py; dihedral importance is only valid for graph
            #    databases rebuilt with that fix. See the caveat near `edge_feature_names`.
            # -------------------------------
            if dihedral_angle_features and edge_attributions.size(1) >= n_edge_features:
                edge_attributions[:, _dihedral_idx] = edge_attributions[:, _dihedral_idx] * dihedral_mask.float()
                edge_attributions[:, _dihedral_idx] = torch.nan_to_num(edge_attributions[:, _dihedral_idx], nan=0.0)

            # Restore model training state
            if was_training:
                task_model.train()

            return node_attributions.detach(), edge_attributions.detach()

        # Use IG in the per-task loop
        ig_node_feature_importance = {t: [] for t in range(5)}
        ig_edge_feature_importance = {t: [] for t in range(5)}
        ig_per_task_impatoms = []
        ig_per_task_preds = []
        ig_per_task_labels = []
        ig_global_smiles = []
        ig_per_task_dfs = []
        # Per-molecule importance arrays collected across all tasks for violin plot
        all_node_importances_per_mol = []
        all_edge_importances_per_mol = []
        all_node_feat_values_per_mol = []
        all_edge_feat_values_per_mol = []

        for task_id in range(5):
            task = task_id
            model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers,
                          n_shared_layers, n_target_specific_layers, input_mode)

            task_model = TaskSpecificGNN(model, task_idx=task, model_args=model_args)
            task_model.eval()
            task_model.to(device)

            # Storage
            smiles_list = []
            predictions = []
            probs = []
            important_atoms_per_mol = []
            correct_val = []
            correct_val_overall = []

            # Loop dataset and compute IG for each molecule
            for i, data in enumerate(testDataset):  # you can limit with [:N] if desired
                data = data.to(device)
                # ensure batch is present and consistent
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

                # Compute model prediction (prob)
                with torch.no_grad():
                    out = task_model(
                        x=data.x,
                        edge_index=data.edge_index,
                        edge_attr=data.edge_attr,
                        batch=data.batch
                    )
                    _prob = out.item()
                    pred = int(_prob > 0.5)
                    predictions.append(pred)
                    probs.append(_prob)

                # Compute IG attributions
                node_attr, edge_attr = integrated_gradients(task_model, data, baseline_x=None, baseline_edge=None,
                                                            steps=50, device=device)
                # node_attr: (N_nodes, F_node), edge_attr: (N_edges, F_edge)

                # Compute importance scores per edge/node (aggregate across feature dims)
                node_scores = node_attr.abs().sum(dim=1).cpu().numpy()  # (N_nodes,)
                edge_scores = edge_attr.abs().sum(dim=1).cpu().numpy()  # (N_edges,)

                # Use same thresholds as before (15% top edges) to pick important edges
                k_edges_tight = max(1, int(0.15 * edge_scores.size))
                top_e_tight = np.argsort(-edge_scores)[:k_edges_tight].tolist()

                imp_edges_tight = data.edge_index[:, torch.tensor(top_e_tight, dtype=torch.long, device=data.edge_index.device)]
                imp_nodes_tight = sorted(set(imp_edges_tight.view(-1).tolist()))

                # Loose filter (30%)
                k_edges_loose = max(1, int(0.30 * edge_scores.size))
                top_e_loose = np.argsort(-edge_scores)[:k_edges_loose].tolist()

                imp_edges_loose = data.edge_index[:, torch.tensor(top_e_loose, dtype=torch.long, device=data.edge_index.device)]
                imp_nodes_loose = sorted(set(imp_edges_loose.view(-1).tolist()))

                # Save raw attributions and input feature values for violin plot
                important_atoms_per_mol.append({
                    "tight_nodes": imp_nodes_tight,
                    "loose_nodes": imp_nodes_loose,
                    "tight_edges": top_e_tight,
                    "loose_edges": top_e_loose,
                    "node_attr": node_attr.cpu().numpy().tolist(),       # IG attributions (N_nodes x F_node)
                    "edge_attr": edge_attr.cpu().numpy().tolist(),       # IG attributions (N_edges x F_edge)
                    "node_feat_vals": data.x.cpu().numpy().tolist(),     # input feature values (N_nodes x F_node)
                    "edge_feat_vals": data.edge_attr.cpu().numpy().tolist(),  # input feature values (N_edges x F_edge)
                    "node_scores": node_scores.tolist(),
                    "edge_scores": edge_scores.tolist()
                })

                # --- Extract SMILES and labels ---
                csv_file = data_file
                df = pd.read_csv(csv_file)
                filepath = os.path.basename(data.file_name)
                molecule_index = int(re.search(r'(\d+)_', filepath).group(1))

                # Resolve column names to support both CSV formats
                _strain_cols = ['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']
                smiles_col = 'SMILES RDKit' if 'SMILES RDKit' in df.columns else 'SMILES'
                row = df.iloc[molecule_index - 1]

                smiles_string = row[smiles_col]
                smiles_list.append(smiles_string)

                correct = row[_strain_cols[task]]
                correct_val.append(correct)

                correct_overall = row['Overall']
                correct_val_overall.append(correct_overall)

            # After looping dataset, store results per task
            ig_per_task_impatoms.append({task_id: important_atoms_per_mol})
            ig_per_task_preds.append({task_id: predictions})
            ig_per_task_labels.append({task_id: correct_val})
            ig_global_smiles = smiles_list

            df = combine(smiles_list, important_atoms_per_mol, predictions, probs, correct_val, correct_val_overall)
            df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
            ig_per_task_dfs.append({task_id: df})

            # --- Compute node/edge feature importance for this task using stored attributions ---
            task_node_importances = []
            task_edge_importances = []

            for mol_idx, row in df.iterrows():
                imp = row["imp_dict"]
                # retrieve raw attributions saved earlier
                node_attr_all = np.array(imp["node_attr"])  # shape (N_nodes, F_node)
                edge_attr_all = np.array(imp["edge_attr"])  # shape (N_edges, F_edge)

                # Select tight nodes and average absolute attributions across selected nodes
                tight_nodes = imp["tight_nodes"]
                if len(tight_nodes) > 0:
                    # Guard: ensure indices in range
                    tight_nodes = [n for n in tight_nodes if n < node_attr_all.shape[0]]
                    if len(tight_nodes) > 0:
                        sel_node_attrs = node_attr_all[tight_nodes]  # (K_nodes, F_node) — keep sign
                        avg_node_feat_importance = sel_node_attrs.mean(axis=0)  # (F_node,)
                        task_node_importances.append(avg_node_feat_importance)
                        all_node_importances_per_mol.append(avg_node_feat_importance)
                        if "node_feat_vals" in imp:
                            node_feat_all = np.array(imp["node_feat_vals"])
                            all_node_feat_values_per_mol.append(node_feat_all[tight_nodes].mean(axis=0))

                # Edge features
                tight_edges = imp["tight_edges"]
                if len(tight_edges) > 0:
                    tight_edges = [e for e in tight_edges if e < edge_attr_all.shape[0]]
                    if len(tight_edges) > 0:
                        sel_edge_attrs = edge_attr_all[tight_edges]  # (K_edges, F_edge) — keep sign
                        avg_edge_feat_importance = sel_edge_attrs.mean(axis=0)
                        task_edge_importances.append(avg_edge_feat_importance)
                        all_edge_importances_per_mol.append(avg_edge_feat_importance)
                        if "edge_feat_vals" in imp:
                            edge_feat_all = np.array(imp["edge_feat_vals"])
                            all_edge_feat_values_per_mol.append(edge_feat_all[tight_edges].mean(axis=0))

            # Aggregate
            if task_node_importances:
                ig_node_feature_importance[task_id] = np.mean(np.vstack(task_node_importances), axis=0)
            if task_edge_importances:
                ig_edge_feature_importance[task_id] = np.mean(np.vstack(task_edge_importances), axis=0)

        # --- Overall feature importance ---
        all_node_importances = []
        all_edge_importances = []

        for t in range(5):
            if isinstance(ig_node_feature_importance[t], np.ndarray):
                all_node_importances.append(ig_node_feature_importance[t])
            if isinstance(ig_edge_feature_importance[t], np.ndarray):
                all_edge_importances.append(ig_edge_feature_importance[t])

        overall_node_importance = np.mean(np.vstack(all_node_importances), axis=0)
        overall_edge_importance = np.mean(np.vstack(all_edge_importances), axis=0)

        # Group RBF distance bins into a single "Distance" value.
        # IG is additive across input dims, so the distance's total attribution is the SUM over
        # its n_dist_feats RBF bins (averaging would divide by n_dist_feats and cancel opposite
        # signs, making distance importance look ~n_dist_feats x too small).
        def _group_edge(arr):
            dist = arr[:n_dist_feats].sum()
            return np.concatenate([[dist], arr[n_dist_feats:]])

        for t in range(5):
            if isinstance(ig_edge_feature_importance[t], np.ndarray):
                ig_edge_feature_importance[t] = _group_edge(ig_edge_feature_importance[t])
        overall_edge_importance = _group_edge(overall_edge_importance)

        node_feature_names = [
            "Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Period 6", "Period 7", "s block", "p block", "d block", "f block",
            "Alkali metals", "Alkaline earth metals", "Transition metals", "Poor metals", "Metalloids", "Nonmetals", "Halogens", "Noble gasses",
            "Lanthanides", "Actinides", "Atomic number", "Atomic radius", "Atomic weight", "Covalent radius", "Density", "Pauling electronegativity",
            "Mass number", "Van der Waals radius"
        ]

        # NOTE on the dihedral feature: a graph-construction bug (dihedral loop scoped outside the
        # edge loop, so only the last edge per molecule got a value) was fixed in XG_graphs.py. The
        # "Dihedral angle" importance below is therefore only meaningful for graph databases REBUILT
        # with the fixed XG_graphs.py (and a model retrained on them). For any older XG database the
        # dihedral column is ~0 for all but one edge per molecule, so its importance there is a data
        # artifact, not a real result. (Bond angle and distance are computed per-edge and are fine.)
        edge_feature_names = (
            ["Distance"] +
            (["Bond angle"] if bond_angle_features else []) +
            (["Dihedral angle"] if dihedral_angle_features else [])
        )

        # Node one-hot feature groupings (shared by the IG violin and the SHAP analysis below).
        node_groups = [
            ("Period", ["Period 1", "Period 2", "Period 3", "Period 4",
                        "Period 5", "Period 6", "Period 7"]),
            ("Block", ["s block", "p block", "d block", "f block"]),
            ("Element group", ["Alkali metals", "Alkaline earth metals",
                               "Transition metals", "Poor metals", "Metalloids",
                               "Nonmetals", "Halogens", "Noble gasses",
                               "Lanthanides", "Actinides"]),
        ]

        # Create output directory for feature importance plots
        plot_dir = os.path.join(args.output_dir, "feature_importance_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Node feature per-task barplots
        plot_task_bars(
            ig_node_feature_importance,
            node_feature_names,
            "Node Feature Importance",
            "node_feature_importance",
            plot_dir
        )

        # Edge feature per-task barplots
        plot_task_bars(
            ig_edge_feature_importance,
            edge_feature_names,
            "Edge Feature Importance",
            "edge_feature_importance",
            plot_dir
        )

        # Overall node features
        plot_overall_bars(
            overall_node_importance,
            node_feature_names,
            "Overall Node Feature Importance",
            "overall_node_feature_importance.png",
            plot_dir
        )

        # Overall edge features
        plot_overall_bars(
            overall_edge_importance,
            edge_feature_names,
            "Overall Edge Feature Importance",
            "overall_edge_feature_importance.png",
            plot_dir
        )

        # Node feature heatmap
        plot_heatmap(
            ig_node_feature_importance,
            node_feature_names,
            "Node Feature Importance per Task",
            "node_feature_importance_heatmap.png",
            plot_dir
        )

        # Edge feature heatmap
        plot_heatmap(
            ig_edge_feature_importance,
            edge_feature_names,
            "Edge Feature Importance per Task",
            "edge_feature_importance_heatmap.png",
            plot_dir
        )

        # SHAP-style scatter plot (node + edge combined)
        if (all_node_importances_per_mol and all_edge_importances_per_mol
                and all_node_feat_values_per_mol and all_edge_feat_values_per_mol
                and len(all_node_feat_values_per_mol) == len(all_node_importances_per_mol)
                and len(all_edge_feat_values_per_mol) == len(all_edge_importances_per_mol)):
            edge_mat = np.vstack(all_edge_importances_per_mol)  # (n_mols, n_edge_features)
            # Sum the RBF-bin attributions (additive IG total) into a single "Distance" attribution.
            dist_attr = edge_mat[:, :n_dist_feats].sum(axis=1, keepdims=True)
            edge_mat_grouped = np.hstack([dist_attr, edge_mat[:, n_dist_feats:]])

            node_feat_mat = np.vstack(all_node_feat_values_per_mol)  # (n_mols, n_node_features)
            edge_feat_mat = np.vstack(all_edge_feat_values_per_mol)  # (n_mols, n_edge_features)
            # Decode an actual distance (Å) from the RBF activations for the colormap. The mean of the
            # bins is ~constant across bonds (a Gaussian bump over densely-spaced centers sums to a
            # near-constant), so it carries no distance info; the activation-weighted center does.
            rbf_params = database_data.get("RBFParameters", {}) or {}
            r_min = float(rbf_params.get("r_min", 0.0))
            r_max = float(rbf_params.get("r_max", 5.0))
            rbf_block = edge_feat_mat[:, :n_dist_feats]
            if n_dist_feats > 1:
                mu = np.linspace(r_min, r_max, n_dist_feats)
                denom = rbf_block.sum(axis=1, keepdims=True)
                denom = np.where(denom == 0, 1.0, denom)
                dist_feat = (rbf_block @ mu).reshape(-1, 1) / denom   # weighted-mean distance, Å
            else:
                dist_feat = rbf_block                                 # raw (non-RBF) distance scalar
            edge_feat_mat_grouped = np.hstack([dist_feat, edge_feat_mat[:, n_dist_feats:]])

            plot_shap_violin(
                np.vstack(all_node_importances_per_mol),
                node_feature_names,
                edge_mat_grouped,
                edge_feature_names,
                node_feat_mat,
                edge_feat_mat_grouped,
                node_groups,
                "Overall Feature Importance (per molecule)",
                "overall_feature_importance_violin.png",
                plot_dir,
            )


if __name__ == "__main__":
    main()




