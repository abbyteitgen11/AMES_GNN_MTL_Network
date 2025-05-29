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
import optuna
from optuna.trial import TrialState

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

from callbacks import set_up_callbacks
from count_model_parameters import count_model_parameters
from device import device, num_workers
from graph_dataset import GraphDataSet
from compute_metrics import *
from data import load_data
from BuildNN_GNN_MTL_HP import BuildNN_GNN_MTL
from masked_loss_function import masked_loss_function
from set_seed import set_seed
from MTLDataset import MTLDataset

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

def objective(trial):
    print(f"Starting trial {trial.number}")
    args = get_args()
    output_dir = ''
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "training.log"))

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Read in input data
    # input_file = sys.argv[1]  # input_file is a yaml compliant file
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
    n_graph_convolution_layers = trial.suggest_int("nGraphConvolutionalLayers", 1,
                                                   5)  # Number of graph convolutional layers
    n_node_neurons = trial.suggest_int("n_node_neurons", 1,
                                       300)  # Number of neurons in GNN
    n_edge_neurons = trial.suggest_int("n_edge_neurons", 1,
                                       300)  # Number of edges in GNN
    dropout_GNN = trial.suggest_float("DropoutGNN", 0.0, 0.5)
    momentum_batch_norm = trial.suggest_float("momentumBatchNorm", 0.0, 1.0)
    n_shared_layers = trial.suggest_int("nSharedLayers", 1, 4) #input_data.get("nSharedLayers", 4)  # Number of layers in shared core
    n_target_specific_layers = trial.suggest_int("nTargetSpecificLayers", 1, 3) #input_data.get("nTargetSpecificLayers", 2)  # Number of layers in target specific core
    n_shared = [
        trial.suggest_int(f"n_shared_{i}", 1, 300)
        for i in range(n_shared_layers)
    ]
    n_target = [
        trial.suggest_int(f"n_target_{i}", 1, 300)
        for i in range(n_target_specific_layers)
    ]
    dropout_shared = [
        trial.suggest_float(f"DropoutShared_{i}", 0.0, 0.5)
        for i in range(n_shared_layers)
    ]
    dropout_target = [
        trial.suggest_float(f"DropoutTarget_{i}", 0.0, 0.5)
        for i in range(n_target_specific_layers)
    ]
    #momentum_batch_norm_shared = trial.suggest_float("momentumBatchNormShared", 0.0, 1.0)
    #momentum_batch_norm_target = trial.suggest_float("momentumBatchNormTarget", 0.0, 1.0)

    activation = input_data.get("ActivationFunction", "ReLU")  # Activation function
    weighted_loss_function = input_data.get("weightedCostFunction", False)
    if weighted_loss_function:
        w1 = trial.suggest_float("w1", 1.0, 6.0)
        w2 = trial.suggest_float("w2", 1.0, 6.0)
        w3 = trial.suggest_float("w3", 1.0, 6.0)
        w4 = trial.suggest_float("w4", 1.0, 6.0)
        w5 = trial.suggest_float("w5", 1.0, 6.0)
        #w1 = 1.90
        #w2 = 1.56
        #w3 = 3.31
        #w4 = 5.09
        #w5 = 5.11
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
    learningRate = trial.suggest_float("learningRate", 1e-5, 1e-1, log=True) #input_data.get("learningRate", 0.0001)  # Learning rate
    weightedCostFunction = input_data.get("weightedCostFunction", None)  # Use weighted  cost function
    L2Regularization = input_data.get("L2Regularization", 0.005)  # L2 regularization coefficient
    loadModel = input_data.get("loadModel", False)
    loadOptimizer = input_data.get("loadOptimizer", False)
    useMolecularDescriptors = input_data.get("useMolecularDescriptors",
                                             False)  # Use molecular descriptors instead of graphs for comparison to original MTL paper

    # Set seeds for consistency
    # set_seed(seed)

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

        # for batch in trainLoader:
        #    print(batch.y.shape)


    else:
        data_path = input_data.get("data_file", "./AMES/data.py")
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

        trainLoader = DataLoader(train_dataset_final, batch_size=nBatch, shuffle=True, generator=g)
        valLoader = DataLoader(val_dataset_final, batch_size=nBatch, generator=g)
        testLoader = DataLoader(test_dataset_final, batch_size=nBatch, generator=g)

    # Build model
    model = BuildNN_GNN_MTL(trial, n_graph_convolution_layers, n_node_neurons, n_edge_neurons, n_node_features, n_edge_features, dropout_GNN, momentum_batch_norm,
                            n_shared_layers, n_target_specific_layers, n_shared, n_target, dropout_shared, dropout_target, activation, useMolecularDescriptors, n_inputs)

    # Write out parameters
    nParameters = count_model_parameters(model)
    log_text += (
            "- This model contains a total of: "
            + repr(nParameters)
            + " adjustable parameters  \n"
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=L2Regularization)  # l2 reg

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

    # move to GPU
    model = model.to(device)

    # Train model
    # factor = float(n_train) / float(n_validation)
    if not useMolecularDescriptors:
        for epoch in range(nEpochs):
            model.train()
            train_loss = 0
            for sample in trainLoader:
                # Compute prediction
                pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                             sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                             n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                             useMolecularDescriptors)
                losses = 0

                for i in range(5):
                    output_key = output_keys[i]
                    loss = masked_loss_function(sample.y[:,i], pred[i].squeeze(1), class_weights[output_key])
                    losses += loss
                loss_final = losses / 5  # Scalar loss

                # Backpropagation
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                train_loss += loss_final.item()

            train_loss /= len(trainLoader)

            # Evaluate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sample in valLoader:
                    pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                                 sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons,
                                 n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                                 useMolecularDescriptors)
                    losses = 0
                    for i in range(5):
                        output_key = output_keys[i]
                        loss = masked_loss_function(sample.y[:, i], pred[i].squeeze(1), class_weights[output_key])
                        losses += loss
                    loss_final = losses / 5  # Scalar loss
                    val_loss += loss_final.item()
            val_loss /= len(valLoader)

            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value
            #if trial.should_prune():
            #    raise optuna.exceptions.TrialPruned()

            # Checkpoints
            if (epoch + 1) % chkptFreq == 0:  # n_epoch + 1 to ensure saving at the last iteration too
                check_point_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    check_point_path,
                )

            # If there are any callbacks, act them if needed
            for callback in callbacks:
                callback(train_loss)
                # check for early stopping; if true, we return to main function
                if (
                        callback.early_stop
                ):  # if we are to stop, make sure we save model/optimizer
                    check_point_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },
                        check_point_path,
                    )

            # Tensorboard
            # if epoch % 10 == 0:
            #    model.eval()
            #    X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)
            #    with torch.no_grad():
            #        y_pred = model(X_internal_tensor.to(device))

            #    metrics_cat = log_metrics(epoch, writer, y_internal, y_pred)
            #    for j, metric_name in enumerate(metric_names):
            #        writer.add_scalar(f"Metrics/{strains}/{metric_name}", metrics_cat[j], epoch)

            if epoch % 10 == 0:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

        writer.close()

        # Make predictions
        y_pred_logit = []
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for sample in valLoader:
                pred = model(sample.x.to(device), sample.edge_index.to(device), sample.edge_attr.to(device),
                             sample.batch.to(device), n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                             n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                             useMolecularDescriptors)
                y_pred_t = tuple(
                    torch.where(tensor > 0.5, torch.tensor(1), torch.tensor(0)) for tensor in pred)  # convert to 0 or 1
                y_pred.append(y_pred_t)
                y_pred_logit.append(pred)
                y_true.append(sample.y)

        y_logit_cat = [np.concatenate([t.numpy() for t in tensors], axis=0) for tensors in
                       zip(*y_pred_logit)]  # concatenate predictions for all examples into single array
        y_logit_cat = np.hstack(y_logit_cat)

        y_pred_cat = [np.concatenate([t.numpy() for t in tensors], axis=0) for tensors in
                      zip(*y_pred)]  # concatenate predictions for all examples into single array
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


    else:
        for epoch in range(nEpochs):
            model.train()
            train_loss = 0
            for X, y in trainLoader:
                # X, y = X.to(device), y.to(device)
                # Compute prediction
                pred = model(X.to(device), 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features,
                             n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                             useMolecularDescriptors)
                # pred = model(X)
                losses = 0

                # print(model.parameters())

                for i in range(5):
                    output_key = output_keys[i]
                    loss = masked_loss_function(y[:, i], pred[i].squeeze(1), class_weights[output_key])
                    losses += loss

                loss_final = losses / 5  # Scalar loss
                # loss_final = losses

                # Backpropagation
                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()

                train_loss += loss_final.item()

            train_loss /= len(trainLoader)

            # Evaluate model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y in valLoader:
                    # X, y = X.to(device), y.to(device)
                    pred = model(X.to(device), 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons,
                                 n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                                 useMolecularDescriptors)
                    # pred = model(X)
                    losses = 0

                    for i in range(5):
                        output_key = output_keys[i]
                        loss = masked_loss_function(y[:, i], pred[i].squeeze(1), class_weights[output_key])
                        losses += loss

                    loss_final = losses / 5  # Scalar loss
                    # loss_final = losses

                    val_loss += loss_final.item()

            val_loss /= len(valLoader)

            trial.report(val_loss, epoch)

            # Handle pruning based on the intermediate value
            #if trial.should_prune():
            #    raise optuna.exceptions.TrialPruned()

            if (epoch + 1) % chkptFreq == 0:  # n_epoch + 1 to ensure saving at the last iteration too
                check_point_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    check_point_path,
                )

            # If there are any callbacks, act them if needed
            for callback in callbacks:
                callback(train_loss)
                # check for early stopping; if true, we return to main function
                if (
                        callback.early_stop
                ):  # if we are to stop, make sure we save model/optimizer
                    check_point_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                        },
                        check_point_path,
                    )

            # if epoch % 10 == 0:
            #    model.eval()
            #    X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)
            #    with torch.no_grad():
            #        y_pred = model(X_internal_tensor.to(device))
            #
            #    metrics_cat = log_metrics(epoch, writer, y_internal, y_pred)
            #    for j, metric_name in enumerate(metric_names):
            #        writer.add_scalar(f"Metrics/{strains}/{metric_name}", metrics_cat[j], epoch)

            if epoch % 10 == 0:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

        writer.close()

        # Make predictions
        model.eval()
        X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)  # .to(device)
        X_external_tensor = torch.tensor(X_external, dtype=torch.float32)
        with torch.no_grad():
            # y_pred = model(X_internal_tensor)
            y_pred = model(X_internal_tensor.to(device), 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons,
                           n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers,
                           useMolecularDescriptors)

        # Convert predictions to numpy arrays
        # y_pred = [yp.cpu().numpy() for yp in y_pred]
        y_pred = [yp.numpy() for yp in y_pred]

        y_pred_98 = np.where(y_pred[0] > 0.5, 1, 0)
        y_pred_100 = np.where(y_pred[1] > 0.5, 1, 0)
        y_pred_102 = np.where(y_pred[2] > 0.5, 1, 0)
        y_pred_1535 = np.where(y_pred[3] > 0.5, 1, 0)
        y_pred_1537 = np.where(y_pred[4] > 0.5, 1, 0)

        # Print to csv
        csv_file = os.path.join(args.output_dir, "metrics.csv")
        headers = ['Strain', 'TP', 'TN', 'FP', 'FN', 'Sp', 'Sn', 'Prec', 'Acc', 'Bal acc', 'F1 score', 'H score']

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(headers)
            _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 0], y_pred_98, y_pred[0])
            metrics = get_metrics(new_real, new_y_pred)
            metrics1 = [int(m) for m in metrics[0]]
            metrics2 = [round(float(m), 2) for m in metrics[1]]
            writer.writerow(['Strain TA98'] + list(metrics1) + list(metrics2))

            _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 1], y_pred_100, y_pred[1])
            metrics = get_metrics(new_real, new_y_pred)
            metrics1 = [int(m) for m in metrics[0]]
            metrics2 = [round(float(m), 2) for m in metrics[1]]
            writer.writerow(['Strain TA100'] + list(metrics1) + list(metrics2))

            _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 2], y_pred_102, y_pred[2])
            metrics = get_metrics(new_real, new_y_pred)
            metrics1 = [int(m) for m in metrics[0]]
            metrics2 = [round(float(m), 2) for m in metrics[1]]
            writer.writerow(['Strain TA102'] + list(metrics1) + list(metrics2))

            _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 3], y_pred_1535, y_pred[3])
            metrics = get_metrics(new_real, new_y_pred)
            metrics1 = [int(m) for m in metrics[0]]
            metrics2 = [round(float(m), 2) for m in metrics[1]]
            writer.writerow(['Strain TA1535'] + list(metrics1) + list(metrics2))

            _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 4], y_pred_1537, y_pred[4])
            metrics = get_metrics(new_real, new_y_pred)
            metrics1 = [int(m) for m in metrics[0]]
            metrics2 = [round(float(m), 2) for m in metrics[1]]
            writer.writerow(['Strain TA1537'] + list(metrics1) + list(metrics2))

            file.flush()
            file.close()

    return val_loss

    # Write to log file
    logging.info(log_text)

    sys.stdout.flush()




#writer.flush()

def save_study(study, path):
    with open(path, "wb") as f:
        pickle.dump(study, f)

def load_study(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    try:
        study = load_study('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')
        print("Loaded existing study.")
    except FileNotFoundError:
        study = optuna.create_study(direction="minimize")
        print("Created new study.")

    study.enqueue_trial({
        "nGraphConvolutionalLayers": 2,
        "n_node_neurons": 78,
        "n_edge_neurons": 107,
        "DropoutGNN": 0.001,
        "momentumBatchNorm": 0.1,
        "nSharedLayers": 4,
        "nTargetSpecificLayers": 2,
        "n_shared_0": 200,
        "n_shared_1": 100,
        "n_shared_2": 50,
        "n_shared_3": 10,
        "n_target_0": 50,
        "n_target_1": 10,
        "DropoutShared_1": 0.25,
        "DropoutShared_2": 0.15,
        "DropoutShared_3": 0.1,
        "DropoutShared_4": 0.0001,
        "DropoutTarget_1": 0.15,
        "DropoutTarget_2": 0.1,
        "w1": 1.90,
        "w2": 1.56,
        "w3": 3.31,
        "w4": 5.09,
        "w5": 5.11,
        "learningRate": 0.0001,
    })

    #study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)

    save_study(study, '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')

    #pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    #print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))