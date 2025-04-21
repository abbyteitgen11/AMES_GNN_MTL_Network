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
from BuildNN_GNN_MTL import BuildNN_GNN_MTL
from masked_loss_function import masked_loss_function
from set_seed import set_seed
from MTLDataset import MTLDataset

# Set seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

#torch.use_deterministic_algorithms(True)
#print(torch.get_rng_state()[:5])

"""
A driver script to fit a Graph Convolutional Neural Network + MTL Neural Network model to
represent properties of molecular/condensed matter systems.

To execute: python GNN_MTL input-file

where input-file is a yaml file specifying different parameters of the
model and how the job is to be run. For an example see sample.yml

"""

# Read in input data
input_file = sys.argv[1]  # input_file is a yaml compliant file

with open( input_file, 'r' ) as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

# Set database path
database_path = input_data.get("database", "./GraphDataBase_AMES")

# The database is described with its own yaml file; so read it
database_file = database_path + '/graph_description.yml'

with open( database_file, 'r' ) as database_stream:
    database_data = yaml.load(database_stream, Loader=yaml.Loader)

# Model parameters
n_node_neurons = input_data.get("nNodeNeurons", 0) # Number of neurons in GNN
n_edge_neurons = input_data.get("nEdgeNeurons", 0) # Number of edges in GNN
n_graph_convolution_layers = input_data.get("nGraphConvolutionLayers", 2) # Number of graph convolutional layers
n_shared_layers = input_data.get("nSharedLayers", 4) # Number of layers in shared core
n_target_specific_layers = input_data.get("nTargetSpecificLayers", 2) # Number of layers in target specific core
n0 = input_data.get("n0", None) # Number of neurons in layer 1 shared core
n1 = input_data.get("n1", None) # Number of neurons in layer 2 shared core
n2 = input_data.get("n2", None) # Number of neurons in layer 3 shared core
n3 = input_data.get("n3", None) # Number of neurons in layer 4 shared core
n4 = input_data.get("n4", None) # Number of neurons in layer 1 target specific core
n5 = input_data.get("n5", None) # Number of neurons in layer 2 target specific core
prob_h1 = input_data.get("prob_h1", None) # Dropout layer 1 shared core
prob_h2 = input_data.get("prob_h2", None) # Dropout layer 2 shared core
prob_h3 = input_data.get("prob_h3", None) # Dropout layer 3 shared core
prob_h4 = input_data.get("prob_h4", None) # Dropout layer 4 shared core
prob_h5 = input_data.get("prob_h5", None) # Dropout layer 1 target specific core
prob_h6 = input_data.get("prob_h6", None) # Dropout layer 2 target specific core
momentum_batch_norm = input_data.get("momentum_batch_norm", None) # Batch normalization
activation = input_data.get("ActivationFunction", "ReLU") # Activation function

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


    # Set up train and val loader
    g = torch.Generator()
    g.manual_seed(seed)
    trainLoader = DataLoader(trainDataset, batch_size=nBatch, num_workers=num_workers, generator=g)
    valLoader = DataLoader(valDataset, batch_size=nBatch, num_workers=num_workers, generator=g)

    #for batch in trainLoader:
    #    print(batch.y.shape)


else:
    data_path = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv'
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

    trainLoader = DataLoader(train_dataset_final, batch_size=nBatch, shuffle = True, generator=g) # , shuffle=True
    valLoader = DataLoader(val_dataset_final, batch_size=nBatch, generator=g)
    testLoader = DataLoader(test_dataset_final, batch_size=nBatch, generator=g)

# File paths for saving model and log
timeString = datetime.now().strftime("%d%m%Y-%H%M%S")
fileName = (
      repr(n_graph_convolution_layers)
    + "gcl-"
    + repr(n_shared_layers)
    + "sl"
    + repr(n_target_specific_layers)
    + "tsl"
    + timeString
    + ".txt"
)
logFile = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs/" + fileName #+ "-" + timeString

saveModelFileName = (
    "GraphPotential-"
    + "nn-"
    + repr(n_graph_convolution_layers)
    + "gcl-"
    + repr(n_shared_layers)
    + "sl"
    + repr(n_target_specific_layers)
    + "tsl"
    + timeString
    + ".tar"
)

check_point_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs/" + saveModelFileName # + "-" + timeString

# Define a Tensorboard writer to monitor the fitting process
writer = SummaryWriter(log_dir='/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs')
#description = markdown.markdown(descriptionText)
#writer.add_text("Description", description)

# molecule = trainDataset[0]
# writer.add_graph( model, [molecule.x, molecule.edge_index,
# molecule.edge_attr] ) # create a graph of model


# Build model
model = BuildNN_GNN_MTL(n0, n1, n2, n3, n4, n5, activation, momentum_batch_norm, prob_h1, prob_h2, prob_h3, prob_h4, prob_h5, prob_h6,
                        n_node_features, n_edge_features, n_node_neurons, n_edge_neurons, n_graph_convolution_layers, n_shared_layers,
                        n_target_specific_layers, useMolecularDescriptors, n_inputs)

# Write out parameters
nParameters = count_model_parameters(model)
log_text += (
    "- This model contains a total of: "
    + repr(nParameters)
    + " adjustable parameters  \n"
)

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

# model to device (cuda or cpu)
#model.to(device)

# Train model
#factor = float(n_train) / float(n_validation)
if not useMolecularDescriptors:
    for epoch in range(nEpochs):
        model.train()
        train_loss = 0
        for sample in trainLoader:
            # Compute prediction
            pred = model(sample.x, sample.edge_index, sample.edge_attr, sample.batch, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
            losses = 0
            #print(model.parameters())

            for i in range(5):
                loss = masked_loss_function(sample.y[:,i], pred[i].squeeze(1))
                losses += loss
            loss_final = losses / 5  # Scalar loss

            # Backpropagation
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            train_loss += loss_final.item()

            # Tensorboard
            #for name, param in model.named_parameters():
                #writer.add_histogram(f"weights/{name}", param, epoch)
                #writer.add_histogram(f"grads/{name}", param.grad, epoch)
                #if param.grad is not None:  # Only log gradients if they exist
                #    writer.add_histogram(f"grads/{name}", param.grad, epoch)
                #else:
                #    print(f"Warning: No gradient for {name}")

        train_loss /= len(trainLoader)

        # Evaluate model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sample in valLoader:
                pred = model(sample.x, sample.edge_index, sample.edge_attr, sample.batch, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
                losses = 0
                for i in range(5):
                    loss = masked_loss_function(sample.y[:, i], pred[i].squeeze(1))
                    losses += loss
                loss_final = losses / 5  # Scalar loss
                val_loss += loss_final.item()
        val_loss /= len(valLoader)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Checkpoints
        if (epoch + 1) % chkptFreq == 0:  # n_epoch + 1 to ensure saving at the last iteration too
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
        #model.eval()
        #X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)
        #with torch.no_grad():
        #    y_pred = model(X_internal_tensor)

        #log_metrics(epoch, writer, y_internal, y_pred)
        if epoch % 5 == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

    # Tensorboard
    #writer.add_histogram("Feature Maps/Layer1", layer1_activations, epoch)
    #for i in range(num_tasks):
    #    writer.add_histogram(f"Task_{i+1}/Predictions", predicted_values[i], epoch)
    #writer.add_scalar(f"Task_{i+1}/Accuracy", accuracy_value, epoch)

    #data_iter = iter(train_loader)
    #x, y = next(data_iter)  # Extract a batch of images and labels
    #images = x.view(x.size(0), -1)  # Flatten if needed
    #writer.add_graph(model, x[:1])  # Use a single sample from the batch

    writer.close()

    # Make predictions
    y_pred_logit = []
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for sample in valLoader:
            pred = model(sample.x, sample.edge_index, sample.edge_attr, sample.batch, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
            y_pred_t = tuple(torch.where(tensor > 0.5, torch.tensor(1), torch.tensor(0)) for tensor in pred) # convert to 0 or 1
            y_pred.append(y_pred_t)
            y_pred_logit.append(pred)
            y_true.append(sample.y)

    y_logit_cat = [np.concatenate([t.numpy() for t in tensors], axis=0) for tensors in zip(*y_pred_logit)] #concatenate predictions for all examples into single array
    y_logit_cat = np.hstack(y_logit_cat)

    y_pred_cat = [np.concatenate([t.numpy() for t in tensors], axis=0) for tensors in zip(*y_pred)] #concatenate predictions for all examples into single array
    y_pred_cat = np.hstack(y_pred_cat)

    y_true_cat = torch.cat(y_true)
    y_true_cat = y_true_cat.numpy()

    # Print out performance info
    print("Strain TA98")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:,0], y_pred_cat[:,0], y_logit_cat[:,0])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA100")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:,1], y_pred_cat[:,1], y_logit_cat[:,1])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA102")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:,2], y_pred_cat[:,2], y_logit_cat[:,2])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA1535")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:,3], y_pred_cat[:,3], y_logit_cat[:,3])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA1537")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_true_cat[:,4], y_pred_cat[:,4], y_logit_cat[:,4])
    print(get_metrics(new_real, new_y_pred))

    # Print to csv
    csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs/performance.csv'
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
    #print(model.linear1.weight[:5, :5])
    #print(torch.get_rng_state()[:5])
    for epoch in range(nEpochs):
        model.train()
        train_loss = 0
        for X, y in trainLoader:
            # X, y = X.to(device), y.to(device)
            # Compute prediction
            pred = model(X, 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
            #pred = model(X)
            losses = 0

            # print(model.parameters())

            for i in range(5):
                loss = masked_loss_function(y[:, i], pred[i].squeeze(1))
                losses += loss

            loss_final = losses / 5  # Scalar loss
            # loss_final = losses

            # Backpropagation
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()

            train_loss += loss_final.item()

            # for name, param in model.named_parameters():
            #    writer.add_histogram(f"weights/{name}", param, epoch)
            ##    writer.add_histogram(f"grads/{name}", param.grad, epoch)
            #    if param.grad is not None:  # Only log gradients if they exist
            #        writer.add_histogram(f"grads/{name}", param.grad, epoch)
            #    #else:
            #        #print(f"Warning: No gradient for {name}")

        train_loss /= len(trainLoader)

        # Evaluate model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in valLoader:
                # X, y = X.to(device), y.to(device)
                pred = model(X, 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)
                #pred = model(X)
                losses = 0

                for i in range(5):
                    loss = masked_loss_function(y[:, i], pred[i].squeeze(1))
                    losses += loss

                loss_final = losses / 5  # Scalar loss
                # loss_final = losses

                val_loss += loss_final.item()

        val_loss /= len(valLoader)

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        if (epoch + 1) % chkptFreq == 0:  # n_epoch + 1 to ensure saving at the last iteration too
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


        # model.eval()
        # X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)  # .to(device)
        # with torch.no_grad():
        #    y_pred = model(X_internal_tensor)

        # log_metrics(epoch, writer, y_internal, y_pred)
        if epoch % 5 == 0:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)

        # writer.add_scalar('Loss/train', train_loss, epoch)
        # writer.add_scalar('Loss/val', val_loss, epoch)

    # Tensorboard
    # writer.add_histogram("Feature Maps/Layer1", layer1_activations, epoch)
    # for i in range(num_tasks):
    #    writer.add_histogram(f"Task_{i+1}/Predictions", predicted_values[i], epoch)
    # writer.add_scalar(f"Task_{i+1}/Accuracy", accuracy_value, epoch)

    # data_iter = iter(train_loader)
    # x, y = next(data_iter)  # Extract a batch of images and labels
    # images = x.view(x.size(0), -1)  # Flatten if needed
    # writer.add_graph(model, x[:1])  # Use a single sample from the batch

    writer.close()

    # Make predictions
    model.eval()
    X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)  # .to(device)
    X_external_tensor = torch.tensor(X_external, dtype=torch.float32)
    with torch.no_grad():
        #y_pred = model(X_internal_tensor)
        y_pred = model(X_internal_tensor, 0, 0, 0, n_node_neurons, n_node_features, n_edge_neurons, n_edge_features, n_graph_convolution_layers, n_shared_layers, n_target_specific_layers, useMolecularDescriptors)

    # Convert predictions to numpy arrays
    y_pred = [yp.numpy() for yp in y_pred]

    y_pred_98 = np.where(y_pred[0] > 0.5, 1, 0)
    y_pred_100 = np.where(y_pred[1] > 0.5, 1, 0)
    y_pred_102 = np.where(y_pred[2] > 0.5, 1, 0)
    y_pred_1535 = np.where(y_pred[3] > 0.5, 1, 0)
    y_pred_1537 = np.where(y_pred[4] > 0.5, 1, 0)

    # Print out performance info
    print("Strain TA98")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 0], y_pred_98, y_pred[0])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA100")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 1], y_pred_100, y_pred[1])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA102")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 2], y_pred_102, y_pred[2])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA1535")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 3], y_pred_1535, y_pred[3])
    print(get_metrics(new_real, new_y_pred))

    print("Strain TA1537")
    print(
        'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
    _, new_real, new_y_pred, new_prob = filter_nan(y_internal[:, 4], y_pred_1537, y_pred[4])
    print(get_metrics(new_real, new_y_pred))

    # Print to csv
    csv_file = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs/performance.csv'
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

# Write to log file
with open(logFile, 'a') as f:
    f.write(log_text)

sys.stdout.flush()

#writer.flush()

# to visualize with tensorboard: tensorboard --logdir='/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/runs'