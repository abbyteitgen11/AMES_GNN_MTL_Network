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

# Load structural alerts as SMARTS
def load_alerts():
    """
    alerts = [
        ("Alkyl esters of phosphonic or sulphonic acids", "C[OX2]P(=O)(O)O or C[OX2]S(=O)(=O)O"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Aromatic N-oxides", "[n+](=O)[O-]"),
        ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Simple aldehydes", "[CX3H1](=O)[#6]"),
        ("N-methylol derivatives", "[NX3]CO"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("S- or N- mustards", "N(CCCl)CCCl or S(CCCl)CCCl"),
        ("Acyl halides", "[CX3](=O)[F,Cl,Br,I]"),
        ("Propiolactones and propiosultones", "O=C1OCC1 or O=S1OCC1"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
        ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
        ("Alkyl nitrites", "[CX4][OX2]N=O"),
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


# Compute overlap between substructure matches and important atoms
def compute_overlap_score(mol, smarts, highlighted_atoms):
    matches = mol.GetSubstructMatches(smarts)
    if not matches:
        return 0.0, []

    highlighted_atoms = set(highlighted_atoms)
    scores = []

    for match in matches:
        match_set = set(match)
        overlap = len(match_set & highlighted_atoms) / len(match_set)
        scores.append(overlap)

    return max(scores), matches

# Compute overlap scores, return df
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

def get_fragment_smiles(mol, atom_indices):
    if mol is None or not atom_indices:
        return None

    n_atoms = mol.GetNumAtoms()
    if any(a >= n_atoms or a < 0 for a in atom_indices):
        #raise ValueError(f"Invalid atom index in {atom_indices} (mol has {n_atoms} atoms)")
        return None

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


def save_fragment_artifacts(df_rows, per_task_sets, frag_examples, output_dir, alerts_compiled, global_smiles, topN_grid=24):
    # Save CSV summary
    df_frags = pd.DataFrame(df_rows)
    frag_csv = os.path.join(output_dir, "explainer_discovered_fragments_summary.csv")
    df_frags.sort_values("total_pos_count", ascending=False).to_csv(frag_csv, index=False)

    # Identify novel candidates: frequent in positives and not matching known alerts
    df_sorted = df_frags.sort_values("total_pos_count", ascending=False)
    novel = df_sorted[(df_sorted["total_pos_count"] >= min(3, max(3, int(len(global_smiles) * 0.005)))) & (
            df_sorted["matched_alerts"] == "")]
    novel.to_csv(os.path.join(output_dir, "explainer_novel_fragment_candidates.csv"), index=False)

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

# Eliminate alerts with < 2 heavy atoms and sort into novel vs non novel
def get_fragment_info_lists(df_rows, alerts_compiled, global_smiles, min_heavy_atoms=2):
    alert_frags, novel_frags = [], []
    for r in df_rows:
        frag = r["fragment"]
        #if not frag:
        #    continue
        #mol = Chem.MolFromSmiles(frag)
        #if mol is None:
        #    continue

        if frag is None:
            return []
        try:
            frag_base = Chem.MolFromSmiles(frag)
            if frag_base is None:
                # Try sanitization: sometimes fragments miss valence info
                try:
                    frag_base = Chem.MolFromSmiles(frag, sanitize=False)
                    Chem.SanitizeMol(frag_base, catchErrors=True)
                except Exception:
                    print(f"[Warning] Could not sanitize invalid fragment: {frag}")
                    return []
            # frag_mol = Chem.AddHs(frag_base)
        except Exception as e:
            print(f"[Error] Could not parse fragment SMILES {frag}: {e}")
            return []

        mol = frag_base

        if mol is None:
            return []


        if mol.GetNumHeavyAtoms() < min_heavy_atoms:
            continue
        if all(a.GetSymbol() in ("C", "H") for a in mol.GetAtoms()):
            continue

        alerts = r["matched_alerts"]
        total_pos_count = r["total_pos_count"]
        total_count = r["total_count"]

        entry = {
            "fragment": frag,
            "mol": mol,
            "alerts": alerts,
            "total_pos_count": total_pos_count,
            "total_count": total_count,
        }

        #if (r["total_pos_count"] >= min(3, max(3, int(len(global_smiles) * 0.005)))) & (
        #        r["matched_alerts"] == ""):
        if r["matched_alerts"] == "":
            novel_frags.append(entry)
        elif alerts:
            alert_frags.append(entry)

    # sort by total_pos_count descending
    alert_frags.sort(key=lambda x: x["total_pos_count"], reverse=True)
    novel_frags.sort(key=lambda x: x["total_pos_count"], reverse=True)
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

        row_imgs = strain_cells + [cons_im, im_alerts_present]
        row_concat = hstack_images(row_imgs, pad=4)

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

def analyze_per_atom_overlap_by_alert(per_task_impatoms, alerts_compiled, global_smiles, per_task_dfs, per_task_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Renamed output directory for clarity that this is POSITIONAL averaging
    plot_dir = os.path.join(output_dir, "alert_averaged_plots_positional")
    os.makedirs(plot_dir, exist_ok=True)
    records = []
    alert_dict = defaultdict(list)
    for name, patt in alerts_compiled:
        alert_dict[name].append(patt)

    ALIPHATIC_ALERTS = {
        "Aliphatic azo and azoxy groups",
        "Aliphatic halogens",
        "Aliphatic N-nitro groups",
        "Alpha, beta unsaturated aliphatic alkoxy groups",
    }

    #EXCLUDED_MOLECULES = {
    #    "Aliphatic azo and azoxy groups": {85},
    #    "Alpha, beta unsaturated aliphatic alkoxy groups": {239, 821, 850},
    #    "Aromatic amines and hydroxylamines": {53, 180}
    #}


    # --- NEW: Store aggregated importance by HASHED FINGERPRINT KEY ---
    # {alert_name: {fp_hash_key: count}}
    environment_importance_freq = defaultdict(lambda: defaultdict(int))
    total_instances_by_alert = defaultdict(int)
    rep_mol_info = {}

    # --- Step 1: Aggregate Environmental Importance Across All Molecules (Hashing Logic) ---
    for alert_name, patt_list in alert_dict.items():

#        skip_ids = EXCLUDED_MOLECULES.get(alert_name, set())

        for task_dict in per_task_impatoms:
            task_id = list(task_dict.keys())[0]
            imp_list = task_dict[task_id]

            for mol_id, smi in enumerate(global_smiles):
                if mol_id >= len(imp_list): continue

 #               if mol_id in skip_ids:
  #                  continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None: continue

                # remove undefined
                per_task_labels_t = per_task_labels[task_id]
                correct_label = int(per_task_labels_t[task_id][mol_id])
                if correct_label == -1:
                    continue

                df0 = per_task_dfs[task_id]
                df_0 = df0[task_id]
                overall_rows = df_0[df_0['mol_id'] == mol_id]
                if overall_rows.empty:
                    continue
                overall_label = int(overall_rows.iloc[0]['label_overall'])
                if overall_label == 0:
                    continue

                max_valid_idx = mol.GetNumAtoms() - 1
                imp_entry = imp_list[mol_id]
                tight_set = set(imp_entry.get("tight", []))

                # --- Index Validation (Guarding against RDKit Range Error) ---
                valid_tight_set = {idx for idx in tight_set if 0 <= idx <= max_valid_idx}

                matched_in_mol = False
                all_match_atoms_in_mol = set()

                valid_matches = []
                for patt in patt_list:
                    matches = mol.GetSubstructMatches(patt)
                    if not matches: continue

                    # --- SAFEGUARD & FILTERING LOGIC ---
                    for match in matches:
                        is_aromatic = False
                        is_in_ring_alkoxy = False

                        # --- SAFEGUARD & FILTERING LOGIC ---
                        for match in matches:

                            if alert_name in ALIPHATIC_ALERTS:
                                # Use the consolidated function for aliphatic rules
                                if is_aliphatic_context_valid(mol, match, alert_name):
                                    valid_matches.append(match)
                            else:
                                # For non-aliphatic alerts, keep all matches
                                valid_matches.append(match)

                    # Process the aggregated valid matches for this molecule
                if valid_matches:

                    matched_in_mol = True
                    patt_smarts = Chem.MolToSmarts(patt_list[0])  # Use the first pattern's SMARTS for context

                    matched_in_mol = True
                    patt_smarts = Chem.MolToSmarts(patt)

                    for match in valid_matches:
                        all_match_atoms_in_mol.update(match)

                if matched_in_mol:

                    # --- NEW FIX: EXPAND THE ALERT ATOM SET FOR COUNTING ---
                    # We create a maximal set of atoms that belong to the structural alert boundary.
                    all_alert_atoms_expanded = set(all_match_atoms_in_mol)

                    # 1. Expand to include all atoms in any ring that contains an alert atom
                    for atom_idx in all_match_atoms_in_mol:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.IsInRing():
                            for ring_idx in range(mol.GetRingInfo().NumRings()):
                                if mol.GetRingInfo().IsAtomInRingOfSize(atom_idx, 0):  # 0 checks all sizes
                                    all_alert_atoms_expanded.update(mol.GetRingInfo().AtomRings()[ring_idx])

                    # 2. Explicitly include all direct heavy-atom neighbors
                    # (Captures O in C=O, or entire N=N=N chain)
                    temp_neighbors = set()
                    for atom_idx in all_match_atoms_in_mol:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetAtomicNum() > 1:  # Only heavy atoms
                                temp_neighbors.add(neighbor.GetIdx())
                    all_alert_atoms_expanded.update(temp_neighbors)
                    # -----------------------------------------------------

                    total_instances_by_alert[alert_name] += 1

                    # --- CORE LOGIC: Hashing and Aggregation (Counting) ---
                    for alert_atom_idx in all_alert_atoms_expanded:

                        try:
                            # Use the RDKit hash of the circular atom environment
                            env_id = AllChem.GetAtomSmi(mol, alert_atom_idx, allHsExplicit=False, isomericSmiles=False)
                            hash_key = env_id

                        except Exception:
                            hash_key = f"atom_{alert_atom_idx}_fail"

                            # Count the hash key if the atom's index was in the GNN's tight set
                        if alert_atom_idx in valid_tight_set:
                            environment_importance_freq[alert_name][hash_key] += 1

                    # Store info about the first molecule found to use as the rep_mol (Visualization Set is now Expanded Set)
                    if alert_name not in rep_mol_info:
                        rep_hash_map = {}
                        for a_idx in all_alert_atoms_expanded:  # Use expanded set for visualization mapping
                            try:
                                env_id = AllChem.GetAtomSmi(mol, a_idx, allHsExplicit=False, isomericSmiles=False)
                                rep_hash_map[a_idx] = env_id
                            except Exception:
                                rep_hash_map[a_idx] = f"atom_{a_idx}_fail"

                        rep_mol_info[alert_name] = {
                            'mol': mol,
                            'rep_hash_map': rep_hash_map,
                            'all_alert_atoms_in_rep_mol': all_alert_atoms_expanded
                            # Store the expanded set for coloring
                        }

    # --- Step 2: Normalize, Map, and Plot Hashed Average (Visualization) ---
    for alert_name, total_instances in total_instances_by_alert.items():
        if total_instances == 0: continue

        env_freq = environment_importance_freq[alert_name]
        rep_data = rep_mol_info.get(alert_name)
        if rep_data is None: continue

        rep_mol = rep_data['mol']
        rep_hash_map = rep_data['rep_hash_map']
        all_alert_atoms_in_rep_mol = rep_data['all_alert_atoms_in_rep_mol']

        # 1. Determine the highest observed frequency (max count) for any environment in this alert.
        max_observed_count = max(env_freq.values()) if env_freq else 1
        max_scaling_factor = max(max_observed_count, 1)

        atom_colors = {}
        highlight_atoms = []
        records = []

        # --- Visualization Logic: Iterate over ALL expanded alert atoms in the representative molecule ---

        for atom_idx in all_alert_atoms_in_rep_mol:

            hash_key = rep_hash_map.get(atom_idx)
            if hash_key is None: continue

            count = env_freq.get(hash_key, 0)

            # 1. Calculate True Linear Score (Count / Total Instances for CSV)
            val_true_freq = count / total_instances

            # 2. Calculate Visual Color Scale (Normalize against MAX OBSERVED COUNT)
            val_color_scale = count / max_scaling_factor

            # 3. Apply Power Scaling (Visual Fix: x^2)
            val_contrast = val_color_scale * val_color_scale

            # 4. Coloring and Highlight (Threshold = 0.1)
            if val_contrast > 0.0025:  # Corresponds to 5% linear frequency

                # Use (1.0, 1.0 - val_contrast, 1.0 - val_contrast) for red scaling (R, G, B)
                R = 1.0
                G = 1.0 - val_contrast
                B = 1.0 - val_contrast

                atom_colors[atom_idx] = (R, G, B)
                highlight_atoms.append(atom_idx)

            # 5. Record keeping for CSV
            records.append({
                "alert": alert_name,
                "atom_index_rep_mol": atom_idx,
                "environment_hash_key": hash_key,
                "importance_freq_normalized": val_true_freq,
                "total_instances": total_instances,
            })

        # Draw the molecule (Code for drawing is assumed to be present and correct)
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        except AttributeError:
            drawer = rdMolDraw2D.MolDraw2D(600, 600)

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            rep_mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightAtomRadii={i: 0.4 for i in highlight_atoms}
        )
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()

        outpath = os.path.join(plot_dir, f"{alert_name.replace('/', '_')}_avg_hashed_env.png")
        with open(outpath, "wb") as fh:
            fh.write(png_bytes)

        # --- Step 3: Call the NEW plotting function for individual instances ---
        plot_important_atoms_by_alert(
            alert_name,
            alert_dict[alert_name],
            global_smiles,
            per_task_impatoms,
            per_task_labels,
            output_dir
        )

    csv_path = os.path.join(output_dir, "alert_atom_overlap_avg_hashed_env_summary.csv")
    pd.DataFrame.from_records(records).to_csv(csv_path, index=False)

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
                imp_atoms_set = mol_task_data[mol_id][task_idx]['tight']

                # Only plot if the GNN found atoms important in this instance
                if imp_atoms_set:
                    if len(mols_to_plot) >= max_mols:
                        break

                    mol_copy = Chem.Mol(mol)
                    atom_colors = {a: (1.0, 0.3, 0.3) for a in imp_atoms_set}

                    per_task_labels_t = per_task_labels[task_idx]
                    correct_label = int(per_task_labels_t[task_idx][mol_id])
                    if correct_label == -1:
                        continue

                    mols_to_plot.append({
                        'mol': mol_copy,
                        'highlight_atoms': list(imp_atoms_set),
                        'atom_colors': atom_colors,
                        'legend': f"Mol ID: {mol_id}, Task: {task_idx + 1}, Smiles: {smi}"
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
    EXCLUDED_MOLECULES = {
        "Aliphatic azo and azoxy groups": {85},
        "Alpha, beta unsaturated aliphatic alkoxy groups": {239, 821, 850},
        "Aromatic amines and hydroxylamines": {53, 180}
    }

    all_alerts = [name for name, _ in alerts_compiled]
    overlap_scores = pd.DataFrame(0.0, index=all_alerts, columns=[f"Strain {i+1}" for i in range(n_tasks)])

    for task in range(n_tasks):
        df_task = per_task_dfs[task][task].copy()

        for alert in all_alerts:
            df_alert = df_task[(df_task["alert"] == alert) & (df_task["alert_present"]) & (df_task["label_overall"] == 1)]
            if len(df_alert) == 0:
                continue

            skip_ids = EXCLUDED_MOLECULES.get(alert, set())

            # 2. Apply the exclusion filter if there are IDs to skip
            if skip_ids:
                # Use the negation (~) of the isin() method to keep only rows NOT in skip_ids
                # We do not need a separate loop; Pandas handles the whole set at once.
                df_alert = df_alert[~df_alert["mol_id"].isin(skip_ids)]

            # Check if any data remains after filtering
            if len(df_alert) == 0:
                mean_overlap = 0.0
            else:
                mean_overlap = df_alert["tight_score"].mean()

            overlap_scores.loc[alert, f"Strain {task + 1}"] = mean_overlap

    mean_overlap_scores = overlap_scores.mean(axis=1)

    return overlap_scores, mean_overlap_scores

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


def main():
    ### Build/load model
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

    ### GNNExplainer analysis
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

        for i, data in enumerate(testDataset[:10]):  # limit if needed for speed
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

        df = evaluate_alerts(smiles_list, important_atoms_per_mol, alerts, predictions, correct_val, correct_val_overall)  # Compute overlap scores by comparing alerts and important nodes, store all in df
        df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
        per_task_dfs.append({task_id: df})

        # Consider an alert present if tight_score > 0 or loose_score > 0
        #df['alert_present'] = (df['tight_score'] > 0) | (df['loose_score'] > 0)
        df['alert_present'] = (df['tight_score'] > 0)

    # Known structural alerts
    alerts_compiled = load_alerts()

    alert_fps = compute_alert_fps(alerts_compiled)

    ### Fragment analysis
    # Which alerts were detected by GNNExplainer, do they overlap with known alerts
    df_rows, per_task_top_sets, frag_examples = build_fragment_catalog(per_task_impatoms, per_task_preds, per_task_labels, global_smiles, alerts_compiled, alert_fps, min_pos_count=3, top_k=200)

    # Save and plot fragment analysis (initial)
    save_fragment_artifacts(df_rows, per_task_top_sets, frag_examples, args.output_dir, alerts_compiled, global_smiles, topN_grid=24)

    # Divide into novel vs not, eliminate alerts with < 2 heavy atoms
    alert_frags, novel_frags = get_fragment_info_lists(df_rows, alerts_compiled, global_smiles, min_heavy_atoms=4)

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

    # Save and plot heatmap of % overlap for each strain
    plot_toxic_overlap_heatmap(overlap_scores, mean_overlap_scores, args.output_dir)

    plot_alert_performance_bars(mean_overlap_scores, args.output_dir)

if __name__ == "__main__":
    main()




