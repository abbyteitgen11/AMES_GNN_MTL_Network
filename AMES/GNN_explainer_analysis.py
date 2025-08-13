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

def extract_submol_from_indices(smiles, atom_indices):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    emol = Chem.EditableMol(Chem.Mol(mol))
    # Remove atoms not in important set
    to_remove = sorted([a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in atom_indices], reverse=True)
    for idx in to_remove:
        emol.RemoveAtom(idx)
    submol = emol.GetMol()
    #Chem.SanitizeMol(submol)
    return submol

def fragment_smiles_from_nodes(smiles, node_idxs):
    mol = Chem.MolFromSmiles(smiles)  # implicit Hs

    if mol is None:
        return None

    n = mol.GetNumAtoms()

    # sanitize and guard indices: cast to int, unique, in-range, sorted
    try:
        idxs = sorted({int(i) for i in node_idxs if 0 <= int(i) < n})
    except Exception:
        return None

    if not idxs:
        return None  # nothing to fragment

    try:
        smi = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=idxs,
            isomericSmiles=False,
            kekuleSmiles=True,
            canonical=True,
            allBondsExplicit=True,
            allHsExplicit=False,
        )
    except Exception:
        return None

    # optional: strip atom maps if present
    smi = re.sub(r'\[\w+:\d+\]', lambda m: m.group(0).split(':')[0] + ']', smi)
    return smi if smi else None



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

    checkpoint = torch.load('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output/checkpoint_epoch_200.pt', map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model = model.to(device)

    substructure_counts_overall = defaultdict(lambda: {"positive": 0, "negative": 0})
    substructure_labels_overall = defaultdict(lambda: {"positive": 0, "negative": 0})

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

            #node_mask = explanation.node_mask.detach().cpu()
            #node_masks_all.append(node_mask.mean(dim=1).numpy())  # importance per atom

            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            k_edges = max(8, int(0.15 * edge_mask.size))  # tune 0.10–0.20; min 8
            top_e = np.argsort(-edge_mask)[:k_edges]

            imp_edges = data.edge_index[:, torch.tensor(top_e, device=data.edge_index.device)]
            imp_nodes = sorted(set(imp_edges.view(-1).tolist()))

            # keep only the largest connected component among these nodes
            G = to_networkx(data, to_undirected=True)
            sub = G.subgraph(imp_nodes).copy()
            if sub.number_of_nodes() > 0:
                lcc = max(nx.connected_components(sub), key=len)
                important_atoms = sorted(list(lcc))
            else:
                important_atoms = []

            important_atoms_per_mol.append(important_atoms)

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

        #threshold = 0.146  # Choose a mask threshold empirically (0–1)

        #for node_mask in node_masks_all:
        #    important = [i for i, val in enumerate(node_mask) if val > threshold]
        #    important_atoms_per_mol.append(important)


        # {substructure: {"positive": count, "negative": count}}
        substructure_counts = defaultdict(lambda: {"positive": 0, "negative": 0})
        substructure_labels = defaultdict(lambda: {"positive": 0, "negative": 0})

        # for each molecule
        for pred, atom_indices, smiles, label, label_overall in zip(predictions, important_atoms_per_mol, smiles_list, correct_val, correct_val_overall):
            smi = fragment_smiles_from_nodes(smiles, atom_indices)
            if not smi:
                continue
            mol_frag = Chem.MolFromSmiles(smi)
            if mol_frag is None:
                continue
            heavy = sum(a.GetAtomicNum() > 1 for a in mol_frag.GetAtoms())
            if not (5 <= heavy <= 18):  # tweak if needed
                continue

            if label != -1:

                if pred == 1:
                    substructure_counts[smi]["positive"] += 1
                elif pred == 0:
                    substructure_counts[smi]["negative"] += 1

                if pred == 1:
                    substructure_counts_overall[smi]["positive"] += 1
                elif pred == 0:
                    substructure_counts_overall[smi]["negative"] += 1

                if label == 1:
                    substructure_labels[smi]["positive"] += 1
                elif label == 0:
                    substructure_labels[smi]["negative"] += 1

                if label == 1:
                    substructure_labels_overall[smi]["positive"] += 1
                elif label == 0:
                    substructure_labels_overall[smi]["negative"] += 1

            #if label_overall != 1:
            #    if label_overall == 1:
            #        substructure_labels_overall[smi]["positive"] += 1
            #    elif label_overall == 0:
            #        substructure_labels_overall[smi]["negative"] += 1


        top_n = 10  # Number of top substructures to show

        # Sort substructures by total frequency
        top_substructures = sorted(
            substructure_counts.items(),
            key=lambda x: x[1]["positive"] + x[1]["negative"],
            reverse=True,
        )

        # Prepare valid entries
        plot_data = []
        for smi, counts in top_substructures:
            if not smi:  # Skip if SMILES string is None or empty
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue  # Skip invalid SMILES

            labels = substructure_labels[smi]

            plot_data.append({
                "mol": mol,
                "smi": smi,
                "positive predictions": counts["positive"],
                "negative predictions": counts["negative"],
                "positive labels": labels["positive"],
                "negative labels": labels["negative"],
            })
            if len(plot_data) == top_n:
                break

        # ---- SAVE PER-TASK SUBSTRUCTURE TABLE ----
        task_rows = []
        for smi, counts in substructure_counts.items():
            labels = substructure_labels[smi]
            task_rows.append({
                "task": task,
                "substructure_smiles": smi,
                "pred_pos": counts["positive"],
                "pred_neg": counts["negative"],
                "label_pos": labels["positive"],
                "label_neg": labels["negative"],
                "total_pred": counts["positive"] + counts["negative"],
                "total_label_defined": labels["positive"] + labels["negative"],
            })
        task_df = pd.DataFrame(task_rows)
        task_csv = os.path.join(args.output_dir, f"substructures_task_{task}.csv")
        task_df.to_csv(task_csv, index=False)
        print(f"Saved per-task substructures to {task_csv}")

        # Plot chart
        fig, axes = plt.subplots(nrows=len(plot_data), ncols=5, figsize=(15, 2 * len(plot_data)))
        fig.suptitle("Top Substructures and Prediction Counts", fontsize=16)

        for i, entry in enumerate(plot_data):
            mol = entry["mol"]
            smi = entry["smi"]
            pos = entry["positive predictions"]
            neg = entry["negative predictions"]
            label_pos = entry["positive labels"]
            label_neg = entry["negative labels"]

            # Molecule image
            img = Draw.MolToImage(mol, size=(200, 200))
            axes[i][0].imshow(img)
            axes[i][0].axis("off")

            # Positive predictions
            axes[i][1].text(0.5, 0.5, str(pos), fontsize=12, ha='center')
            axes[i][1].set_title("Positive Predictions")
            axes[i][1].axis("off")

            # Negative predictons
            axes[i][2].text(0.5, 0.5, str(neg), fontsize=12, ha='center')
            axes[i][2].set_title("Negative Predictions")
            axes[i][2].axis("off")

            # Positive labels
            axes[i][3].text(0.5, 0.5, str(label_pos), fontsize=12, ha='center')
            axes[i][3].set_title("Positive Labels")
            axes[i][3].axis("off")

            # Negative labels
            axes[i][4].text(0.5, 0.5, str(label_neg), fontsize=12, ha='center')
            axes[i][4].set_title("Negative Labels")
            axes[i][4].axis("off")

        #plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # ---- SAVE OVERALL SUBSTRUCTURE TABLE ----
    overall_rows = []
    for smi, counts in substructure_counts_overall.items():
        labels = substructure_labels_overall[smi]
        overall_rows.append({
            "substructure_smiles": smi,
            "pred_pos": counts["positive"],
            "pred_neg": counts["negative"],
            "label_pos": labels["positive"],
            "label_neg": labels["negative"],
            "total_pred": counts["positive"] + counts["negative"],
            "total_label_defined": labels["positive"] + labels["negative"],
        })
    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(args.output_dir, "substructures_overall.csv")
    overall_df.to_csv(overall_csv, index=False)
    print(f"Saved overall substructures to {overall_csv}")

    # Plot top substructures overall
    top_n = 10
    top_substructures_overall = sorted(
        substructure_counts_overall.items(),
        key=lambda x: x[1]["positive"] + x[1]["negative"],
        reverse=True
    )

    plot_data = []
    for smi, counts in top_substructures_overall:
        if not smi:  # Skip if SMILES string is None or empty
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # Skip invalid SMILES

        labels_overall = substructure_labels_overall[smi]

        plot_data.append({
            "mol": mol,
            "smi": smi,
            "positive predictions": counts["positive"],
            "negative predictions": counts["negative"],
            "positive labels": labels_overall["positive"],
            "negative labels": labels_overall["negative"],
        })
        if len(plot_data) == top_n:
            break

    fig, axes = plt.subplots(nrows=len(plot_data), ncols=5, figsize=(15, 2 * len(plot_data)))
    fig.suptitle("Top 10 Substructures (Overall)", fontsize=16)

    for i, entry in enumerate(plot_data):
        mol = entry["mol"]
        smi = entry["smi"]
        pos = entry["positive predictions"]
        neg = entry["negative predictions"]
        label_pos = entry["positive labels"]
        label_neg = entry["negative labels"]

        # Molecule image
        img = Draw.MolToImage(mol, size=(200, 200))
        axes[i][0].imshow(img)
        axes[i][0].axis("off")

        # SMILES
        #axes[i][1].text(0, 0.5, smi, fontsize=9, wrap=True)
        #axes[i][1].axis("off")

        # Positive predictions
        axes[i][1].text(0.5, 0.5, str(pos), fontsize=12, ha='center')
        axes[i][1].set_title("Positive Predictions")
        axes[i][1].axis("off")

        # Negative predictions
        axes[i][2].text(0.5, 0.5, str(neg), fontsize=12, ha='center')
        axes[i][2].set_title("Negative Predictions")
        axes[i][2].axis("off")

        # Positive labels
        axes[i][3].text(0.5, 0.5, str(label_pos), fontsize=12, ha='center')
        axes[i][3].set_title("Positive Labels")
        axes[i][3].axis("off")

        # Negative labels
        axes[i][4].text(0.5, 0.5, str(label_neg), fontsize=12, ha='center')
        axes[i][4].set_title("Negative Labels")
        axes[i][4].axis("off")

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

    