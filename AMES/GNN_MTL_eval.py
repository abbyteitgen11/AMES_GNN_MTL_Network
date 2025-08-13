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
from torch_geometric.utils import to_networkx

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


"""
A driver script to fit a Graph Convolutional Neural Network + MTL Neural Network model to
represent properties of molecular/condensed matter systems.

To execute: python GNN_MTL input-file

where input-file is a yaml file specifying different parameters of the
model and how the job is to be run. For an example see sample.yml

"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    return parser.parse_args()


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to compute and log metrics
def log_metrics(epoch, writer, y_internal, y_pred):
    strains = ["TA98", "TA100", "TA102", "TA1535", "TA1537"]
    metric_names = ["TP", "TN", "FP", "FN", "Specificity", "Sensitivity", "Precision", "Accuracy", "Balanced Accuracy", "F1 Score", "H Score"]

    y_pred_out = [yp.cpu().numpy() for yp in y_pred]

    y_pred_out[0] = np.where(y_pred_out[0] > 0.5, 1, 0)
    y_pred_out[1] = np.where(y_pred_out[1] > 0.5, 1, 0)
    y_pred_out[2] = np.where(y_pred_out[2] > 0.5, 1, 0)
    y_pred_out[3] = np.where(y_pred_out[3] > 0.5, 1, 0)
    y_pred_out[4] = np.where(y_pred_out[4] > 0.5, 1, 0)

    for i, strains in enumerate(strains):
        _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, i], y_pred_out[i], y_pred[i])
        metrics = get_metrics(new_real, new_y_pred)
        metrics_cat = np.concatenate(metrics)

    return metrics_cat


def visualize_model_parameters(model):
    # Collect parameter info
    param_data = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            module = name.split('.')[0]  # top-level module name
            param_data.append((module, param.numel()))

    # Create DataFrame
    df = pd.DataFrame(param_data, columns=["Module", "Count"])
    summary = df.groupby("Module")["Count"].sum().sort_values()

    # Plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(summary.index, summary.values)
    plt.xlabel("Number of Parameters")

    # Add labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(summary.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f'{width:,}',
                 va='center')

    plt.tight_layout()
    plt.show()


def consensus_from_heads(y_pred_cat):
    # Your exact OR rule with -1 if mixed
    N = y_pred_cat.shape[0]
    y_cons = np.zeros(N, dtype=int)
    for i in range(N):
        row = y_pred_cat[i]
        if np.any(row == 1):
            y_cons[i] = 1
        elif np.all(row == 0):
            y_cons[i] = 0
        else:
            y_cons[i] = -1
    return y_cons

def consensus_truth(y_true_cat):
    N = y_true_cat.shape[0]
    y_cons_true = np.zeros(N, dtype=int)
    for i in range(N):
        row = y_true_cat[i]
        if np.any(row == 1):
            y_cons_true[i] = 1
        elif np.all(row == 0):
            y_cons_true[i] = 0
        else:
            y_cons_true[i] = -1
    return y_cons_true

def eval_consensus_metric(y_true_cat, y_logit_cat, thresholds, metric="balanced_accuracy"):
    """Apply thresholds -> consensus preds, then return your metric on valid rows."""
    probs = y_logit_cat  # (N,5)
    y_pred_cat = (probs >= np.array(thresholds)[None, :]).astype(int)  # (N,5)
    y_cons_pred = consensus_from_heads(y_pred_cat)
    y_cons_true = consensus_truth(y_true_cat)

    # Use your existing masking + metrics
    _, new_real, new_y_pred, _ = filter_nan(y_cons_true, y_cons_pred, y_cons_pred)
    # get_metrics returns (counts, scores)
    counts, scores = get_metrics(new_real, new_y_pred)
    # scores order (per your code): Sp, Sn, Prec, Acc, BalAcc, F1, H
    sp, sn, prec, acc, balacc, f1, h = scores
    return (sp if metric == "sp" else h), (sp, sn, prec, acc, balacc, f1, h)


import numpy as np

def one_se_choice(th_grid, scores):
    """
    Pick a threshold within 1 standard error of the best score.
    Return the median of eligible thresholds (more stable than the single best).
    """
    scores = np.array(scores, dtype=float)
    best = np.max(scores)
    se = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    eligible = [t for t, s in zip(th_grid, scores) if s >= best - se]
    return float(np.median(eligible)) if eligible else float(th_grid[int(np.argmax(scores))])

def coord_ascent_consensus(y_true_cat, y_prob_cat, init_th=None, metric="sn", rounds=3):
    """
    Coordinate ascent over the 5 thresholds with the 1-SE rule per coordinate.
    y_prob_cat: (N,5) probabilities in [0,1] (your y_logit_cat as you use it now)
    metric: "sn" (your current choice), or swap to "balanced_accuracy" / "h" if you prefer.
    """
    ths = [0.5]*5 if init_th is None else list(init_th)
    grid = np.linspace(0.05, 0.95, 19)  # coarse but robust

    best_val, best_scores = eval_consensus_metric(y_true_cat, y_prob_cat, ths, metric)

    for _ in range(rounds):
        improved = False
        for h in range(5):
            # Evaluate metric across grid while holding other thresholds fixed
            scores = []
            for t in grid:
                trial = ths.copy()
                trial[h] = float(t)
                val, _ = eval_consensus_metric(y_true_cat, y_prob_cat, trial, metric)
                scores.append(val)
            # Choose threshold using 1-SE rule
            t_star = one_se_choice(grid, scores)
            trial = ths.copy(); trial[h] = t_star
            val, scr = eval_consensus_metric(y_true_cat, y_prob_cat, trial, metric)
            if val > best_val + 1e-9:
                ths[h] = float(t_star)
                best_val, best_scores = val, scr
                improved = True
        if not improved:
            break
    return ths, best_val, best_scores

def crossfit_thresholds_for_consensus(y_true_cat, y_prob_cat, K=5, metric="sn", seed=0):
    """
    K-fold cross-fit on the *validation* set: for each fold,
    learn thresholds on K-1 folds with coord-ascent+1SE; aggregate (median) across folds.
    Returns list of 5 thresholds.
    """
    N = len(y_true_cat)
    idx = np.arange(N)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, K)

    ths_per_fold = []
    for k in range(K):
        val_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        ths_k, _, _ = coord_ascent_consensus(y_true_cat[tr_idx], y_prob_cat[tr_idx], metric=metric)
        ths_per_fold.append(ths_k)

    ths_per_fold = np.array(ths_per_fold)  # (K,5)
    ths_final = np.median(ths_per_fold, axis=0).tolist()
    return ths_final



def main():
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "training.log"))

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Read in input data
    #input_file = sys.argv[1]  # input_file is a yaml compliant file
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
        # }
        class_weights = {
            '98': {0: 0.801, 1: 1.330, -1: 0.0},
            '100': {0: 0.885, 1: 1.149, -1: 0.0},
            '102': {0: 0.692, 1: 1.799, -1: 0.0},
            '1535': {0: 0.604, 1: 2.907, -1: 0.0},
            '1537': {0: 0.602, 1: 2.941, -1: 0.0},
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
    GNN_explainer_analysis = input_data.get("GNNExplainerAnalysis", False) # Run GNNExplainer analysis after training

    # Set seeds for consistency
    #set_seed(seed)

    # Print out model information to log
    log_text = "\n# Model Description  \n"
    log_text += "- nGraphConvolutionLayers: " + repr(n_graph_convolution_layers) + "  \n"
    log_text += "- nSharedLayers: " + repr(n_shared_layers) + "  \n"
    log_text += "- nTargetSpecificLayers: " + repr(n_target_specific_layers) + "  \n"

    log_text += "- nEpochs: " + repr(nEpochs) + "  \n"
    log_text += "- nBatch: " + repr(nBatch) + "  \n"
    log_text += (
        "- Checkpointing model and optimizer every " + repr(chkptFreq) + " epochs  \n"
    )
    log_text += "- learningRate: " + repr(learningRate) + "  \n"
    log_text += "- random seed: " + repr(seed) + "  \n"

    log_text += "- graph construction style: " + graph_type + "  \n"

    if nTrainMaxEntries:
        log_text += "- Size of training database: " + repr(nTrainMaxEntries) + "  \n"
    else:
        log_text += "- Using full training database \n"
    if nValMaxEntries:
        log_text += "- Size of validation/test database: " + repr(nValMaxEntries) + "  \n"
    else:
        log_text += "- Using full validation/test database  \n"

    if loadModel:
        loadModelFileName = input_data["StateDictFileName"]
        log_text += "- Starting from previous model: " + loadModelFileName + "  \n"
    else:
        log_text += "- Initialising model parameters from scratch   \n"

    descriptionText = input_data.get("descriptionText", " ")
    descriptionText += log_text

    if not useMolecularDescriptors:
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

        # Set up train and val loader
        trainLoader = DataLoader(trainDataset, batch_size=nBatch, generator=g)
        valLoader = DataLoader(valDataset, batch_size=nBatch, generator=g)
        testLoader = DataLoader(testDataset, batch_size=nBatch, generator=g)

        #for batch in trainLoader:
        #    print(batch.y.shape)


    else:
        data_path = input_data.get("data_file", "./AMES/data.csv")
        train, internal, external = load_data(data_path, model="MTL", stage='GS')
        X_train, y_train = train
        X_internal, y_internal = internal
        X_external, y_external = external

        # Reformat data
        X_train = X_train[:, 1:]  # Remove SMILES
        X_internal = X_internal[:, 1:]  # Remove SMILES
        X_external = X_external[:, 1:]  # Remove SMILES

        y_train = np.transpose(y_train)
        y_internal = np.transpose(y_internal)
        y_external = np.transpose(y_external)

        # Convert to float
        X_train = np.array(X_train, dtype=np.float32)
        X_internal = np.array(X_internal, dtype=np.float32)
        X_external = np.array(X_external, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_internal = np.array(y_internal, dtype=np.float32)
        y_external = np.array(y_external, dtype=np.float32)

        if nTrainMaxEntries:
            X_train = X_train[:nTrainMaxEntries]
            Y_train = y_train[:nTrainMaxEntries]
        if nValMaxEntries:
            X_internal = X_internal[:nValMaxEntries]
            y_internal = y_internal[:nValMaxEntries]

        # Convert to tensor
        train_dataset = torch.tensor(X_train, dtype=torch.float32)
        train_output = torch.tensor(y_train, dtype=torch.float32)
        val_dataset = torch.tensor(X_internal, dtype=torch.float32)
        val_output = torch.tensor(y_internal, dtype=torch.float32)
        test_dataset = torch.tensor(X_external, dtype=torch.float32)
        test_output = torch.tensor(y_external, dtype=torch.float32)


        # Convert to dataset
        train_dataset_final = MTLDataset(train_dataset, train_output)
        val_dataset_final = MTLDataset(val_dataset, val_output)
        test_dataset_final = MTLDataset(test_dataset, test_output)

        n_inputs = X_train.shape[1]

        g = torch.Generator()
        g.manual_seed(seed)

        trainLoader = DataLoader(train_dataset_final, batch_size=nBatch, shuffle = True, generator=g)
        valLoader = DataLoader(val_dataset_final, batch_size=nBatch, generator=g)
        testLoader = DataLoader(test_dataset_final, batch_size=nBatch, generator=g)


    # Build model
    model = BuildNN_GNN_MTL(n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features, n_edge_features, dropout_GNN, momentum_batch_norm,
                            n_shared_layers, n_target_specific_layers, n_shared, n_target, dropout_shared, dropout_target,
                            activation, useMolecularDescriptors, n_inputs)

    # Write out parameters
    nParameters = count_model_parameters(model)
    log_text += (
        "- This model contains a total of: "
        + repr(nParameters)
        + " adjustable parameters  \n"
    )

    # Example usage:
    print("Total trainable parameters:", count_trainable_parameters(model))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"'{name}': {param.numel()},")

    #visualize_model_parameters(model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization) # l2 reg

    # If we are to use a pre-saved model and optimizer, load their parameters here
    if loadModel:
        checkpoint = torch.load(loadModelFileName)
        model.load_state_dict(checkpoint["model_state_dict"])

        if loadOptimizer:
           optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        n_start = checkpoint["epoch"]

    else:
        n_start = 0

    # setup callbacks, if any
    anyCallBacks = input_data.get("callbacks", None)
    callbacks = set_up_callbacks(anyCallBacks, optimizer)


    checkpoint = torch.load('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output/checkpoint_epoch_200.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    #optimizer.load_state_dict(checkspoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

    #thresholds = [0.55, 0.55, 0.45, 0.48, 0.48]
    # Compute thresholds

    y_pred_logit = []
    y_true = []
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(valLoader):
            pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                         sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                         n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
            y_pred_logit.append(pred)
            y_true.append(sample.y)

    y_logit_cat = [np.concatenate([t.cpu().numpy() for t in tensors], axis=0) for tensors in
                   zip(*y_pred_logit)]  # concatenate predictions for all examples into single array
    y_logit_cat = np.hstack(y_logit_cat)

    y_true_cat = torch.cat(y_true)
    y_true_cat = y_true_cat.numpy()

    best_ths = crossfit_thresholds_for_consensus(
        y_true_cat=y_true_cat,
        y_prob_cat=y_logit_cat,
        K=5,
        metric="sp",  # or "balanced_accuracy" / "h"
        seed=42
    )
    print("Cross-fit consensus thresholds:", best_ths)

    # (Optional) see val performance at the cross-fit thresholds
    val_metric, val_scores = eval_consensus_metric(y_true_cat, y_logit_cat, best_ths, metric="sp")
    print("Validation consensus scores (Sp, Sn, Prec, Acc, BalAcc, F1, H):", val_scores)

    thresholds =  [0.5, 0.5, 0.7749999999999999, 0.5, 0.5]
    #thresholds = [0.5, 0.5, 0.5, 0.5, 0.5]

    # Make predictions
    y_pred_logit = []
    y_pred = []
    y_true = []
    file_names = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(testLoader):
            pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device), sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
            y_pred_t = tuple(torch.where(tensor > thresholds[i], torch.tensor(1), torch.tensor(0)) for i, tensor in enumerate(pred))
            #y_pred_t = tuple(torch.where(tensor > 0.5, torch.tensor(1), torch.tensor(0)) for tensor in pred) # convert to 0 or 1
            y_pred.append(y_pred_t)
            y_pred_logit.append(pred)
            y_true.append(sample.y)

            for data in sample.to_data_list():  # unpack the batch into individual graphs
                file_names.append(data.file_name)

    y_logit_cat = [np.concatenate([t.cpu().numpy() for t in tensors], axis=0) for tensors in zip(*y_pred_logit)] #concatenate predictions for all examples into single array
    y_logit_cat = np.hstack(y_logit_cat)

    y_pred_cat = [np.concatenate([t.cpu().numpy() for t in tensors], axis=0) for tensors in zip(*y_pred)] #concatenate predictions for all examples into single array
    y_pred_cat = np.hstack(y_pred_cat)

    y_true_cat = torch.cat(y_true)
    y_true_cat = y_true_cat.numpy()


    # Print to csv
    csv_file = os.path.join(args.output_dir, "metrics.csv")
    headers = ['Strain', 'TP', 'TN', 'FP', 'FN', 'Sp', 'Sn', 'Prec', 'Acc', 'Bal acc', 'F1 score', 'H score']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(headers)
        _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:, 0], y_pred_cat[:, 0], y_logit_cat[:, 0])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA98'] + list(metrics1) + list(metrics2))

        _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:, 1], y_pred_cat[:, 1], y_logit_cat[:, 1])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA100'] + list(metrics1) + list(metrics2))

        _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:, 2], y_pred_cat[:, 2], y_logit_cat[:, 2])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA102'] + list(metrics1) + list(metrics2))

        _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:, 3], y_pred_cat[:, 3], y_logit_cat[:, 3])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA1535'] + list(metrics1) + list(metrics2))

        _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:, 4], y_pred_cat[:, 4], y_logit_cat[:, 4])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA1537'] + list(metrics1) + list(metrics2))

        file.flush()
        file.close()

    # Write to log file
    logging.info(log_text)

    sys.stdout.flush()


    data_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv"

    df = pd.read_csv(data_path)
    y_overall = df['Overall'].values
    #partition = df['Partition']
    index = []

    for file_path in file_names:
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d+)_', filename)
        first_number = int(match.group(1))
        index.append(first_number-1)

    y_labels_overall = y_overall[index]
    #print(partition[index])

    #overall = load_data(data_path, model="Overall", stage='EVAL')
    #y_overall = overall[1]

    y_cons = np.zeros(len(y_true_cat[:,0]))
    y_cons_true = np.zeros(len(y_true_cat[:,0]))

    for i in range(len(y_true_cat[:,0])):
        if y_pred_cat[i, 0] == 1 or y_pred_cat[i, 1] == 1 or y_pred_cat[i, 2] == 1 or y_pred_cat[i, 3] == 1 or y_pred_cat[i, 4] == 1:
            y_cons[i] = 1
        elif y_pred_cat[i, 0] == 0 and y_pred_cat[i, 1] == 0 and y_pred_cat[i, 2] == 0 and y_pred_cat[i, 3] == 0 and y_pred_cat[i, 4] == 0:
            y_cons[i] = 0
        else:
            y_cons[i] = -1

    for i in range(len(y_true_cat[:,0])):
        if y_true_cat[i, 0] == 1 or y_true_cat[i, 1] == 1 or y_true_cat[i, 2] == 1 or y_true_cat[i, 3] == 1 or y_true_cat[i, 4] == 1:
            y_cons_true[i] = 1
        elif y_true_cat[i, 0] == 0 and y_true_cat[i, 1] == 0 and y_true_cat[i, 2] == 0 and y_true_cat[i, 3] == 0 and y_true_cat[i, 4] == 0:
            y_cons_true[i] = 0
        else:
            y_cons_true[i] = -1

    wrong_indices = np.where(y_cons != y_cons_true)[0]
    wrong_files = [file_names[i] for i in wrong_indices]

    df_wrong = pd.DataFrame({
        'file_name': wrong_files,
        'true_label': [y_cons_true[i] for i in wrong_indices],
        'pred_label': [y_cons[i] for i in wrong_indices],
     })

    df_wrong.to_csv(os.path.join(args.output_dir, "misclassified_files.csv"), index=False)

    # Print to csv
    csv_file = os.path.join(args.output_dir, "metrics_cons.csv")
    headers = ['Strain', 'TP', 'TN', 'FP', 'FN', 'Sp', 'Sn', 'Prec', 'Acc', 'Bal acc', 'F1 score', 'H score']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(headers)
        _, new_real, new_y_pred, new_prob = filter_nan(y_cons_true, y_cons, y_logit_cat[:, 0])
        metrics = get_metrics(new_real, new_y_pred)
        metrics1 = [int(m) for m in metrics[0]]
        metrics2 = [round(float(m), 2) for m in metrics[1]]
        writer.writerow(['Strain TA98'] + list(metrics1) + list(metrics2))

        file.flush()
        file.close()



    csv_file = os.path.join(args.output_dir, "model_output_raw.csv")
    headers = ['file', 'logits_98', 'logits_100', 'logits_102', 'logits_1535', 'logits_1537','y_true_98', 'y_true_100', 'y_true_102', 'y_true_1535', 'y_true_1537', 'y_pred_98', 'y_pred_100', 'y_pred_102', 'y_pred_1535', 'y_pred_1537', 'y_true_consensus', 'y_pred_consensus']

    data_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv"
    external = load_data(data_path, model="MTL", stage='EVAL')
    x_test = external[0]
    x_test_labels = x_test[:, 0]

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(headers)

        #smiles_array = np.array(x_test_labels).flatten()
        file_names_array = np.array(file_names).flatten()
        logits_98_array = np.array(y_logit_cat[:, 0]).astype(float).flatten()
        logits_100_array = np.array(y_logit_cat[:, 1]).astype(float).flatten()
        logits_102_array = np.array(y_logit_cat[:, 2]).astype(float).flatten()
        logits_1535_array = np.array(y_logit_cat[:, 3]).astype(float).flatten()
        logits_1537_array = np.array(y_logit_cat[:, 4]).astype(float).flatten()
        y_true_98_array = np.array(y_true_cat[:, 0]).astype(float).flatten()
        y_true_100_array = np.array(y_true_cat[:, 1]).astype(float).flatten()
        y_true_102_array = np.array(y_true_cat[:, 2]).astype(float).flatten()
        y_true_1535_array = np.array(y_true_cat[:, 3]).astype(float).flatten()
        y_true_1537_array = np.array(y_true_cat[:, 4]).astype(float).flatten()
        y_pred_98_array = np.array(y_pred_cat[:, 0]).astype(float).flatten()
        y_pred_100_array = np.array(y_pred_cat[:, 1]).astype(float).flatten()
        y_pred_102_array = np.array(y_pred_cat[:, 2]).astype(float).flatten()
        y_pred_1535_array = np.array(y_pred_cat[:, 3]).astype(float).flatten()
        y_pred_1537_array = np.array(y_pred_cat[:, 4]).astype(float).flatten()
        y_true_consensus_array = np.array(y_labels_overall).astype(float).flatten()
        y_pred_consensus_array = np.array(y_cons).astype(float).flatten()

        rows = zip(
            file_names_array,
            logits_98_array, logits_100_array, logits_102_array, logits_1535_array, logits_1537_array,
            y_true_98_array, y_true_100_array, y_true_102_array, y_true_1535_array, y_true_1537_array,
            y_pred_98_array, y_pred_100_array, y_pred_102_array, y_pred_1535_array, y_pred_1537_array,
            y_true_consensus_array, y_pred_consensus_array
        )

        writer.writerows(rows)

    # Write to log file
    logging.info(log_text)

    sys.stdout.flush()


#writer.flush()top_substructures = sorted(substructure_counts.items(), key=lambda x: x[1]["positive"] + x[1]["negative"], reverse=True)[:20]

#mols = [Chem.MolFromSmiles(smi) for smi, _ in top_substructures]
#legends = [f"+: {counts['positive']}, -: {counts['negative']}" for _, counts in top_substructures]

#img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(250,250))
#img.show()top_substructures = sorted(substructure_counts.items(), key=lambda x: x[1]["positive"] + x[1]["negative"], reverse=True)[:20]

#mols = [Chem.MolFromSmiles(smi) for smi, _ in top_substructures]
#legends = [f"+: {counts['positive']}, -: {counts['negative']}" for _, counts in top_substructures]

#img = Draw.MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(250,250))
#img.show()

if __name__ == "__main__":
    main()