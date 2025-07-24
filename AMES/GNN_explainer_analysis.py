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
from BuildNN_GNN_MTL_global import BuildNN_GNN_MTL
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
        class_weights = {
            '98': {0: 1.0, 1: w1, -1: 0},
            '100': {0: 1.0, 1: w2, -1: 0},
            '102': {0: 1.0, 1: w3, -1: 0},
            '1535': {0: 1.0, 1: w4, -1: 0},
            '1537': {0: 1.0, 1: w5, -1: 0},
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

    checkpoint = torch.load('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/checkpoints/checkpoint_epoch_1000.pt', map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    model = model.to(device)

    substructure_counts_overall = defaultdict(lambda: {"positive": 0, "negative": 0})
    substructure_labels_overall = defaultdict(lambda: {"positive": 0, "negative": 0, "undefined": 0})

    for task_id in range(5):

        task = task_id
        model_args = (n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)

        task_model = TaskSpecificGNN(model, task_idx=task, model_args=model_args)
        task_model.eval()

        explainer = Explainer(
            model=task_model,
            algorithm=GNNExplainer(epochs=50),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
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
                global_feats=data.global_feats
            )

            with torch.no_grad():
                task_output = task_model(
                    x=data.x,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    batch=data.batch,
                    global_feats=data.global_feats
                )

                prediction = int(task_output.item() > 0.5)  # 1 = toxic, 0 = non-toxic
                predictions.append(prediction)

            node_mask = explanation.node_mask.detach().cpu()
            node_masks_all.append(node_mask.mean(dim=1).numpy())  # importance per atom

            # Extract SMILES
            # CSV file with structure data
            csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/DataBase_AMES/FILES/ames_mutagenicity_data.csv'
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

        threshold = 0.146  # Choose a mask threshold empirically (0–1)

        for node_mask in node_masks_all:
            important = [i for i, val in enumerate(node_mask) if val > threshold]
            important_atoms_per_mol.append(important)

        # {substructure: {"positive": count, "negative": count}}
        substructure_counts = defaultdict(lambda: {"positive": 0, "negative": 0})
        substructure_labels = defaultdict(lambda: {"positive": 0, "negative": 0, "undefined": 0})

        # for each molecule
        for pred, atom_indices, smiles, label, label_overall in zip(predictions, important_atoms_per_mol, smiles_list, correct_val, correct_val_overall):
            submol = extract_submol_from_indices(smiles, atom_indices)
            smi = Chem.MolToSmiles(submol, canonical=True)
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
            elif label == -1:
                substructure_labels[smi]["undefined"] += 1

            if label_overall == 1:
                substructure_labels_overall[smi]["positive"] += 1
            elif label_overall == 0:
                substructure_labels_overall[smi]["negative"] += 1
            elif label_overall == -1:
                substructure_labels_overall[smi]["undefined"] += 1

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
                "undefined labels": labels["undefined"],
            })
            if len(plot_data) == top_n:
                break

        # Plot chart
        fig, axes = plt.subplots(nrows=len(plot_data), ncols=6, figsize=(15, 2 * len(plot_data)))
        fig.suptitle("Top Substructures and Prediction Counts", fontsize=16)

        for i, entry in enumerate(plot_data):
            mol = entry["mol"]
            smi = entry["smi"]
            pos = entry["positive predictions"]
            neg = entry["negative predictions"]
            label_pos = entry["positive labels"]
            label_neg = entry["negative labels"]
            label_undef = entry["undefined labels"]

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

            # Undefined labels
            axes[i][5].text(0.5, 0.5, str(label_undef), fontsize=12, ha='center')
            axes[i][5].set_title("Undefined Labels")
            axes[i][5].axis("off")

        #plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

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
            "undefined labels": labels_overall["undefined"],
        })
        if len(plot_data) == top_n:
            break

    fig, axes = plt.subplots(nrows=len(plot_data), ncols=6, figsize=(15, 2 * len(plot_data)))
    fig.suptitle("Top 10 Substructures (Overall)", fontsize=16)

    for i, entry in enumerate(plot_data):
        mol = entry["mol"]
        smi = entry["smi"]
        pos = entry["positive predictions"]
        neg = entry["negative predictions"]
        label_pos = entry["positive labels"]
        label_neg = entry["negative labels"]
        label_undef = entry["undefined labels"]

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

        # Undefined labels
        axes[i][5].text(0.5, 0.5, str(label_undef), fontsize=12, ha='center')
        axes[i][5].set_title("Undefined Labels")
        axes[i][5].axis("off")

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()

    