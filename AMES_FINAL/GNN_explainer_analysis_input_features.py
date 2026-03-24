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


def combine(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall):
    rows = []

    for i, (smiles, imp_dict, pred, label, label_overall) in enumerate(
            zip(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall)):
        # mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        mol = Chem.MolFromSmiles(smiles)

        rows.append({
            "mol_id": i,
            "smiles": smiles,
            "imp_dict": imp_dict,
            "prediction": pred,
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
        xticklabels=feature_names,             # ✔ correct seaborn argument
        yticklabels=[f"Task {t}" for t in task_ids],  # ✔ correct seaborn argument
    )

    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Tasks")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()

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

    """
    ### GNNExplainer analysis
    # To store per-task feature importance
    node_feature_importance = {t: [] for t in range(5)}
    edge_feature_importance = {t: [] for t in range(5)}

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
            important_atoms_tight = imp_nodes_tight

            # Loose filter
            k_edges_loose = int(0.15 * edge_mask.size)  # max(8, int(0.15 * edge_mask.size))  # ~25–30%
            top_e_loose = np.argsort(-edge_mask)[:k_edges_loose]

            imp_edges_loose = data.edge_index[:, torch.tensor(top_e_loose, device=data.edge_index.device)]
            imp_nodes_loose = sorted(set(imp_edges_loose.view(-1).tolist()))

            sub_loose = G.subgraph(imp_nodes_loose).copy()
            if sub_loose.number_of_nodes() > 0:
                comps = max(nx.connected_components(sub_loose), key=len)
                important_atoms_loose = sorted(list(comps))
            else:
                important_atoms_loose = []

            # Collect both sets
            important_atoms_per_mol.append({
                "tight_nodes": important_atoms_tight,
                "loose_nodes": important_atoms_loose,
                "tight_edges": top_e_tight.tolist(),
                "loose_edges": top_e_loose.tolist(),
                "edge_mask": edge_mask.tolist()
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

        df = combine(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall)  # Compute overlap scores by comparing alerts and important nodes, store all in df
        df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
        per_task_dfs.append({task_id: df})

        # --- Compute node feature importance for this task ---
        task_node_importances = []
        task_edge_importances = []

        for mol_idx, row in df.iterrows():
            imp_nodes = row["imp_dict"]["tight_nodes"]  # or "loose", or combine both
            data = testDataset[mol_idx].to(device)

            # NODE FEATURES
            if len(imp_nodes) > 0:
                node_feats = data.x[imp_nodes]  # shape = (#imp_nodes, num_features)
                avg_node_feat_importance = node_feats.abs().mean(dim=0).cpu().numpy()
                task_node_importances.append(avg_node_feat_importance)

            # EDGE FEATURES
            imp = row["imp_dict"]

            top_edges = imp["tight_edges"]  # or "loose_edges"

            if len(top_edges) > 0:
                edge_feats = data.edge_attr[top_edges]  # Safe now
                avg_edge_feat_importance = edge_feats.abs().mean(dim=0).cpu().numpy()
                task_edge_importances.append(avg_edge_feat_importance)

        # Aggregate feature importance for this task
        if task_node_importances:
            node_feature_importance[task_id] = np.mean(np.vstack(task_node_importances), axis=0)

        if task_edge_importances:
            edge_feature_importance[task_id] = np.mean(np.vstack(task_edge_importances), axis=0)
            
    """

    # -----------------------------
    # Replace the GNNExplainer section with Integrated Gradients
    # -----------------------------

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
        # dihedral = edge_attr[:, 2]
        dihedral_mask = ~torch.isnan(edge_attr[:, 2])  # True where valid

        # -------------------------------
        # 3. Create a clean version (NaNs -> 0)
        # -------------------------------
        clean_edge_attr = edge_attr.clone()
        clean_edge_attr[:, 2] = torch.nan_to_num(clean_edge_attr[:, 2], nan=0.0)

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

        #-------------------------------
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
        #    Set attribution = 0 for edges where dihedral doesn't exist
        # -------------------------------
        if edge_attributions.size(1) > 2:
            edge_attributions[:, 2] = edge_attributions[:, 2] * dihedral_mask.float()
            edge_attributions[:, 2] = torch.nan_to_num(edge_attributions[:, 2], nan=0.0)

        # Restore model training state
        if was_training:
            task_model.train()

        return node_attributions.detach(), edge_attributions.detach()

    # Use IG in the per-task loop
    # Clear previous importance dicts
    node_feature_importance = {t: [] for t in range(5)}
    edge_feature_importance = {t: [] for t in range(5)}

    for task_id in range(5):
        task = task_id
        model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers,
                      n_shared_layers, n_target_specific_layers, useMolecularDescriptors)

        task_model = TaskSpecificGNN(model, task_idx=task, model_args=model_args)
        task_model.eval()
        task_model.to(device)

        # Storage
        smiles_list = []
        predictions = []
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
                pred = int(out.item() > 0.5)
                predictions.append(pred)

            #df = pd.DataFrame({
            #    "dihedral": data.edge_attr[:, 2].cpu().numpy(),
            #    #"label": labels.cpu().numpy()
            #})

            #print(df.corr())

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

            # Loose filter (same fraction; you could change to larger fraction if desired)
            k_edges_loose = max(1,
                                int(0.30 * edge_scores.size))  # make loose a bit larger (30%) — adjust if you want same 15%
            top_e_loose = np.argsort(-edge_scores)[:k_edges_loose].tolist()

            imp_edges_loose = data.edge_index[:, torch.tensor(top_e_loose, dtype=torch.long, device=data.edge_index.device)]
            imp_nodes_loose = sorted(set(imp_edges_loose.view(-1).tolist()))

            # Save raw attributions too (so later you can compute feature-wise averages)
            important_atoms_per_mol.append({
                "tight_nodes": imp_nodes_tight,
                "loose_nodes": imp_nodes_loose,
                "tight_edges": top_e_tight,
                "loose_edges": top_e_loose,
                "node_attr": node_attr.cpu().numpy().tolist(),  # store raw attributions (N_nodes x F_node)
                "edge_attr": edge_attr.cpu().numpy().tolist(),  # store raw attributions (N_edges x F_edge)
                "node_scores": node_scores.tolist(),
                "edge_scores": edge_scores.tolist()
            })

            # --- Extract SMILES and labels (same as your existing code) ---
            csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv'
            df = pd.read_csv(csv_file)
            filepath = os.path.basename(data.file_name)
            molecule_index = int(re.search(r'(\d+)_', filepath).group(1))
            smiles_column_index = 3
            correct_val_index = 1364 + task
            correct_val_overall_index = 1369

            smiles_string = df.iloc[molecule_index - 1, smiles_column_index]
            smiles_list.append(smiles_string)

            correct = df.iloc[molecule_index - 1, correct_val_index]
            correct_val.append(correct)

            correct_overall = df.iloc[molecule_index - 1, correct_val_overall_index]
            correct_val_overall.append(correct_overall)

        # After looping dataset, store results per task (matching your previous format)
        per_task_impatoms.append({task_id: important_atoms_per_mol})
        per_task_preds.append({task_id: predictions})
        per_task_labels.append({task_id: correct_val})
        global_smiles = smiles_list

        df = combine(smiles_list, important_atoms_per_mol, predictions, correct_val, correct_val_overall)
        df = df[(df["label"] != -1) & (df["label_overall"] != -1)]
        per_task_dfs.append({task_id: df})

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
                    sel_node_attrs = np.abs(node_attr_all[tight_nodes])  # (K_nodes, F_node)
                    avg_node_feat_importance = sel_node_attrs.mean(axis=0)  # (F_node,)
                    task_node_importances.append(avg_node_feat_importance)

            # Edge features
            tight_edges = imp["tight_edges"]
            if len(tight_edges) > 0:
                tight_edges = [e for e in tight_edges if e < edge_attr_all.shape[0]]
                if len(tight_edges) > 0:
                    sel_edge_attrs = np.abs(edge_attr_all[tight_edges])  # (K_edges, F_edge)
                    avg_edge_feat_importance = sel_edge_attrs.mean(axis=0)
                    task_edge_importances.append(avg_edge_feat_importance)

        # Aggregate
        if task_node_importances:
            node_feature_importance[task_id] = np.mean(np.vstack(task_node_importances), axis=0)
        if task_edge_importances:
            edge_feature_importance[task_id] = np.mean(np.vstack(task_edge_importances), axis=0)

    # --- Overall feature importance ---
    all_node_importances = []
    all_edge_importances = []

    for t in range(5):
        if isinstance(node_feature_importance[t], np.ndarray):
            all_node_importances.append(node_feature_importance[t])
        if isinstance(edge_feature_importance[t], np.ndarray):
            all_edge_importances.append(edge_feature_importance[t])

    overall_node_importance = np.mean(np.vstack(all_node_importances), axis=0)
    overall_edge_importance = np.mean(np.vstack(all_edge_importances), axis=0)

    node_feature_names = [
        "Period 1", "Period 2", "Period 3", "Period 4", "Period 5", "Period 6", "Period 7", "s block", "p block", "d block", "f block",
        "Alkali metals", "Alkaline earth metals", "Transition metals", "Poor metals", "Metalloids", "Nonmetals", "Halogens", "Noble gasses",
        "Lanthanides", "Actinides", "Atomic number", "Atomic radius", "Atomic weight", "Covalent radius", "Density", "Pauling electronegativity",
        "Mass number", "Van der Waals radius"
    ]

    edge_feature_names = ["Distance", "Bond angle", "Dihedral angle"]

    # -----------------------------
    # Create output directory
    # -----------------------------
    plot_dir = os.path.join(args.output_dir, "feature_importance_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Node feature per-task barplots
    plot_task_bars(
        node_feature_importance,
        node_feature_names,
        "Node Feature Importance",
        "node_feature_importance",
        plot_dir
    )

    # Edge feature per-task barplots
    plot_task_bars(
        edge_feature_importance,
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
        node_feature_importance,
        node_feature_names,
        "Node Feature Importance per Task",
        "node_feature_importance_heatmap.png",
        plot_dir
    )

    # Edge feature heatmap
    plot_heatmap(
        edge_feature_importance,
        edge_feature_names,
        "Edge Feature Importance per Task",
        "edge_feature_importance_heatmap.png",
        plot_dir
    )


if __name__ == "__main__":
    main()






