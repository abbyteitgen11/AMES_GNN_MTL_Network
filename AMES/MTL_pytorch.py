
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import csv

import h5py
import sys

from compute_metrics import *
from data import load_data

# --------------------------------------------------------------------------------

def set_seed(s):
    seed_value = s

    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # Set the `pytorch` pseudo-random generator at a fixed value
    torch.manual_seed(seed_value)

    # If using CUDA, set the seed for all GPUs
    #torch.cuda.manual_seed(seed_value)
    #torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups

    # Ensure deterministic behavior (slower but fully reproducible)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------

def masked_loss_function(y_true, y_pred):
    mask_value = -1

    # Create mask (1 for valid values, 0 for masked values)
    mask = (y_true != mask_value).float()

    y_true_clamped = torch.clamp(y_true, min=0, max=1)

    # Compute BCE loss for all elements
    loss = F.binary_cross_entropy(y_pred, y_true_clamped, reduction='none')  # Keep per-element loss

    # Apply mask (zero out masked values)
    loss = loss * mask

    #loss_final = loss.sum() / mask.sum()

    loss_final = loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0) # If NaN, set to 0

    return loss_final


# --------------------------------------------------------------------------------
# Load data
data_path = 'data.csv'
train, internal = load_data(data_path, model="MTL", stage="GS")
X_train, y_train = train
X_internal, y_internal = internal

# Reformat data
X_train = X_train[:, 1:]  # Remove SMILES
X_internal = X_internal[:, 1:]  # Remove SMILES

y_train = np.transpose(y_train)
y_internal = np.transpose(y_internal)

# Convert to float
X_train = np.array(X_train, dtype=np.float32)
X_internal = np.array(X_internal, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_internal = np.array(y_internal, dtype=np.float32)

# Convert to tensor
train_dataset = torch.tensor(X_train, dtype=torch.float32)
train_output = torch.tensor(y_train, dtype=torch.float32)
val_dataset = torch.tensor(X_internal, dtype=torch.float32)
val_output = torch.tensor(y_internal, dtype=torch.float32)


class MTLDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return feature and target pair for each example
        return self.features[idx], self.targets[idx]


# Convert to dataset
train_dataset_final = MTLDataset(train_dataset, train_output)
val_dataset_final = MTLDataset(val_dataset, val_output)

# Parameters
# Dropout
prob_h1 = 0.25
prob_h2 = 0.15
prob_h3 = 0.1
prob_h4 = 0.0001

# Batch normalization
momentum_batch_norm = 0.9

# Activation function
act = "ReLU"

# Size of input layer
n_inputs = X_train.shape[1]

# Number of epochs
n_epochs = 1 #np.iinfo(np.int32).max

# Number of neurons shared core (layers 1, 2, 3, 4)
n0, n1, n2, n3 = (200, 100, 50, 10)

# Weighted cost function
w = None

# Number of layers in target specific core
spec_lay = 2

# random seed
seed_r = 0
set_seed(seed_r)

# L2 regularization coefficient
lb = 0.005

# Learning rate
l_rate = 0.0005 #0.0001

# Early stopping
min_delta_val = 0.0005
patience_val = 1000


#params = 'lb_' + str(lb) + '_spec_lay_' + str(spec_lay) + '_n0_' + str(n0) + '_n1_' + str(n1) + '_n2_' + str(
            #n2) + '_n3_' + str(n3) + '_w_' + str(w)

# for monitoring purposes
#print(params)
#sys.stdout.flush()

# Callbacks *move to separate file
class EarlyStopping:
    def __init__(self, patience=patience_val, min_delta=min_delta_val, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.counter = 0
        self.best_weights = None

    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True  # Stop training
        return False

early_stop = EarlyStopping()

# Print out true and pred values every epoch *move to separate file
class PrintPredictionsCallback:
    def __init__(self, data, csv_filename="predictions_Torch.csv"):
        self.data = data
        self.csv_filename = csv_filename

        with open(self.csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Task", "True Value", "Predicted Value", "Training Loss", "Validation Loss"])

    def on_epoch_end(self, epoch, model, train_loss, val_loss):
        model.eval()
        X_sample, y_sample = self.data
        X_sample = torch.tensor(X_sample, dtype=torch.float32)  # Convert to tensor
        y_sample = torch.tensor(y_sample, dtype=torch.float32)  # Convert to tensor

        with torch.no_grad():
            preds = model(X_sample)

        # Print the epoch number
        #print(f"\nEpoch {epoch + 1}:")
        rows = []

        # Iterate through each task
        for i in range(5):
            true_values = y_sample[:, i][:10].cpu().numpy()
            predicted_values = preds[i][:10].cpu().numpy()

            for true_val, pred_val in zip(true_values, predicted_values):
                rows.append([epoch + 1, i + 1, float(true_val), float(pred_val), float(train_loss), float(val_loss)])

            #print(f"  Task {i + 1} - True: {y_sample[:, i][:10].cpu().numpy()}, Pred: {preds[i][:10].cpu().numpy()}")

        #print(f"Epoch {epoch + 1} - Loss: {loss.item()}")
        with open(self.csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)


# Create the callback with your data
print_callback = PrintPredictionsCallback((X_internal, y_internal))

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

        # Log each metric to TensorBoard
        for j, metric_name in enumerate(metric_names):
            writer.add_scalar(f"Metrics/{strains}/{metric_name}", metrics_cat[j], epoch)


class BuildNN(nn.Module):
    def __init__(self,
                ni: int = 34,
                n0: int = 200,
                n1: int = 100,
                n2: int = 50,
                n3: int = 10,
                act: str = "ReLu",
                momentum_batch_norm: float = 0.9,
                spec_layers: int = 2,
                prob_h1: float = 0.25,
                prob_h2: float = 0.15,
                prob_h3: float = 0.1,
                prob_h4: float = 0.0001):
        super(BuildNN, self).__init__()

        # Shared core
        self.activation_layer = nn.ReLU()
        self.linear1 = nn.Linear(ni, n0)
        #self.linear1.requires_grad = True
        self.bn1 = nn.BatchNorm1d(n0, momentum=momentum_batch_norm)
        self.dropout1 = nn.Dropout(prob_h1)
        self.linear2 = nn.Linear(n0, n1)
        #self.linear2.requires_grad = True
        self.bn2 = nn.BatchNorm1d(n1, momentum=momentum_batch_norm)
        self.dropout2 = nn.Dropout(prob_h2)
        self.linear3 = nn.Linear(n1, n2)
        #self.linear3.requires_grad = True
        self.bn3 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.dropout3 = nn.Dropout(prob_h3)
        self.linear4 = nn.Linear(n2, n3)
        #self.linear4.requires_grad = True
        self.bn4 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.dropout4 = nn.Dropout(prob_h4)

        # Target specific core
        self.ts1_linear1 = nn.Linear(n3, n2) # x.size(1)
        #self.ts1_linear1.requires_grad = True
        self.ts1_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts1_dropout1 = nn.Dropout(prob_h2)
        self.ts1_linear2 = nn.Linear(n2, n3)
        #self.ts1_linear2.requires_grad = True
        self.ts1_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts1_dropout2 = nn.Dropout(prob_h3)
        self.ts1_sig = nn.Linear(n3,1)

        self.ts2_linear1 = nn.Linear(n3, n2)  # x.size(1)
        #self.ts2_linear1.requires_grad = True
        self.ts2_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts2_dropout1 = nn.Dropout(prob_h2)
        self.ts2_linear2 = nn.Linear(n2, n3)
        #self.ts2_linear2.requires_grad = True
        self.ts2_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts2_dropout2 = nn.Dropout(prob_h3)
        self.ts2_sig = nn.Linear(n3, 1)

        self.ts3_linear1 = nn.Linear(n3, n2)  # x.size(1)
        #self.ts3_linear1.requires_grad = True
        self.ts3_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts3_dropout1 = nn.Dropout(prob_h2)
        self.ts3_linear2 = nn.Linear(n2, n3)
        #self.ts3_linear2.requires_grad = True
        self.ts3_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts3_dropout2 = nn.Dropout(prob_h3)
        self.ts3_sig = nn.Linear(n3, 1)

        self.ts4_linear1 = nn.Linear(n3, n2)  # x.size(1)
        #self.ts4_linear1.requires_grad = True
        self.ts4_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts4_dropout1 = nn.Dropout(prob_h2)
        self.ts4_linear2 = nn.Linear(n2, n3)
        #self.ts4_linear2.requires_grad = True
        self.ts4_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts4_dropout2 = nn.Dropout(prob_h3)
        self.ts4_sig = nn.Linear(n3, 1)

        self.ts5_linear1 = nn.Linear(n3, n2)  # x.size(1)
        #self.ts5_linear1.requires_grad = True
        self.ts5_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts5_dropout1 = nn.Dropout(prob_h2)
        self.ts5_linear2 = nn.Linear(n2, n3)
        #self.ts5_linear2.requires_grad = True
        self.ts5_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts5_dropout2 = nn.Dropout(prob_h3)
        self.ts5_sig = nn.Linear(n3, 1)


    def forward(self, x):
        # Shared core
        x = self.dropout1(x)
        x = self.linear1(x)
        #x = x.relu()
        x = self.activation_layer(self.bn1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        #x = x.relu()
        x = self.activation_layer(self.bn2(x))
        x = self.dropout3(x)
        x = self.linear3(x)
        #x = x.relu()
        x = self.activation_layer(self.bn3(x))
        x = self.dropout4(x)
        x = self.linear4(x)
        #x = x.relu()
        x = self.activation_layer(self.bn4(x))

        # Target specific core
        # Two layers
        # STRAIN 1
        y1 = self.ts1_dropout1(x)
        y1 = self.ts1_linear1(y1)
        #y1 = y1.relu()
        y1 = self.activation_layer(self.ts1_bn1(y1))
        y1 = self.ts1_dropout2(y1)
        y1 = self.ts1_linear2(y1)
        #y1 = y1.relu()
        y1 = self.activation_layer(self.ts1_bn2(y1))
        y1 = self.ts1_sig(y1)
        y1 = y1.sigmoid()

        y2 = self.ts2_dropout1(x)
        y2 = self.ts2_linear1(y2)
        #y2 = y2.relu()
        y2 = self.activation_layer(self.ts2_bn1(y2))
        y2 = self.ts2_dropout2(y2)
        y2 = self.ts2_linear2(y2)
        #y2 = y2.relu()
        y2 = self.activation_layer(self.ts2_bn2(y2))
        y2 = self.ts2_sig(y2)
        y2 = y2.sigmoid()

        y3 = self.ts3_dropout1(x)
        y3 = self.ts3_linear1(y3)
        # y3 = y3.relu()
        y3 = self.activation_layer(self.ts3_bn1(y3))
        y3 = self.ts3_dropout2(y3)
        y3 = self.ts3_linear2(y3)
        # y3 = y3.relu()
        y3 = self.activation_layer(self.ts3_bn2(y3))
        y3 = self.ts3_sig(y3)
        y3 = y3.sigmoid()

        y4 = self.ts4_dropout1(x)
        y4 = self.ts4_linear1(y4)
        #y4 = y4.relu()
        y4 = self.activation_layer(self.ts4_bn1(y4))
        y4 = self.ts4_dropout2(y4)
        y4 = self.ts4_linear2(y4)
        #y4 = y4.relu()
        y4 = self.activation_layer(self.ts4_bn2(y4))
        y4 = self.ts4_sig(y4)
        y4 = y4.sigmoid()

        y5 = self.ts5_dropout1(x)
        y5 = self.ts5_linear1(y5)
        #y5 = y5.relu()
        y5 = self.activation_layer(self.ts5_bn1(y5))
        y5 = self.ts5_dropout2(y5)
        y5 = self.ts5_linear2(y5)
        #y5 = y5.relu()
        y5 = self.activation_layer(self.ts5_bn2(y5))
        y5 = self.ts5_sig(y5)
        y5 = y5.sigmoid()

        return y1, y2, y3, y4, y5

# Build MTL_DNN model
model = BuildNN(1348, 200, 100, 50, 10, "ReLu", 0.9, 2, 0.25, 0.15, 0.1, 0.0001)
#model = BuildNN(n_inputs, n0, n1, n2, n3, reg, act, spec_lay, momentum_batch_norm)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=lb) # replace l2 reg


# Create DataLoader
train_loader = DataLoader(train_dataset_final, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset_final, batch_size=32)


for X, y in train_loader:
    print(f"Shape of X: {X.shape}, type of X: {type(X)}")
    print(f"Shape of y: {y.shape}, type of y: {type(y)}")
    break

print(f"Length of train dataloader: {len(train_loader)} batches of 32")
print(f"Length of test dataloader: {len(val_loader)} batches of 32")

#model.to(device)

#writer = SummaryWriter('runs/MTL_pytorch')

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        #X, y = X.to(device), y.to(device)

        # Compute prediction
        pred = model(X)
        losses = 0

        #print(model.parameters())

        for i in range(5):
            loss = masked_loss_function(y[:, i], pred[i].squeeze(1))
            losses += loss

        loss_final = losses / 5  # Scalar loss
        #loss_final = losses

        # Backpropagation
        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()

        train_loss += loss_final.item()

        #for name, param in model.named_parameters():
        #    writer.add_histogram(f"weights/{name}", param, epoch)
        ##    writer.add_histogram(f"grads/{name}", param.grad, epoch)
        #    if param.grad is not None:  # Only log gradients if they exist
        #        writer.add_histogram(f"grads/{name}", param.grad, epoch)
        #    #else:
        #        #print(f"Warning: No gradient for {name}")

    train_loss /= len(train_loader)


    # Evaluate model on test dataset
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            #X, y = X.to(device), y.to(device)

            pred = model(X)
            losses = 0

            for i in range(5):
                loss = masked_loss_function(y[:, i], pred[i].squeeze(1))
                losses += loss

            loss_final = losses / 5  # Scalar loss
            #loss_final = losses

            val_loss += loss_final.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Check early stopping
    if early_stop(model, val_loss):
        print("Early stopping triggered.")
        break

    # Call custom callback
    #print_callback.on_epoch_end(epoch, model, train_loss, val_loss)

    #model.eval()
    #X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32)  # .to(device)
    #with torch.no_grad():
    #    y_pred = model(X_internal_tensor)

    #log_metrics(epoch, writer, y_internal, y_pred)

    #writer.add_scalar('Loss/train', train_loss, epoch)
    #writer.add_scalar('Loss/val', val_loss, epoch)



#writer.add_histogram("Feature Maps/Layer1", layer1_activations, epoch)
#for i in range(num_tasks):
#    writer.add_histogram(f"Task_{i+1}/Predictions", predicted_values[i], epoch)
#writer.add_scalar(f"Task_{i+1}/Accuracy", accuracy_value, epoch)

#data_iter = iter(train_loader)
#x, y = next(data_iter)  # Extract a batch of images and labels
#images = x.view(x.size(0), -1)  # Flatten if needed
#writer.add_graph(model, x[:1])  # Use a single sample from the batch


#writer.close()


# Make predictions
model.eval()
X_internal_tensor = torch.tensor(X_internal, dtype=torch.float32) #.to(device)
with torch.no_grad():
    y_pred = model(X_internal_tensor)

# Convert predictions to numpy arrays
y_pred = [yp.cpu().numpy() for yp in y_pred]

y_pred_98 = np.where(y_pred[0] > 0.5, 1, 0)
y_pred_100 = np.where(y_pred[1] > 0.5, 1, 0)
y_pred_102 = np.where(y_pred[2] > 0.5, 1, 0)
y_pred_1535 = np.where(y_pred[3] > 0.5, 1, 0)
y_pred_1537 = np.where(y_pred[4] > 0.5, 1, 0)

# Print out performance info
print("Strain TA98")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[:,0], y_pred_98, y_pred[0])
print(get_metrics(new_real, new_y_pred))

print("Strain TA100")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[:,1], y_pred_100, y_pred[1])
print(get_metrics(new_real, new_y_pred))

print("Strain TA102")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[:,2], y_pred_102, y_pred[2])
print(get_metrics(new_real, new_y_pred))

print("Strain TA1535")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[:,3], y_pred_1535, y_pred[3])
print(get_metrics(new_real, new_y_pred))

print("Strain TA1537")
print(
    'internal validation: (TP, TN, FP, FN) (Sp --- Sn --- Prec --- Acc --- Bal acc --- F1 score --- H score --- AUC)')
_, new_real, new_y_pred, new_prob = filter_nan(y_internal[:,4], y_pred_1537, y_pred[4])
print(get_metrics(new_real, new_y_pred))


sys.stdout.flush()
# Clear memory
#K.clear_session()
