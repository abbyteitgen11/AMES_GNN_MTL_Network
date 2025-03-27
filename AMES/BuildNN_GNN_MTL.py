import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool


class BuildNN_GNN_MTL(nn.Module):
    def __init__(self,
                 ni: int = 168,
                 n0: int = 200,
                 n1: int = 100,
                 n2: int = 50,
                 n3: int = 10,
                 act: str = "ReLu",
                 momentum_batch_norm: float = 0.9,
                 prob_h1: float = 0.25,
                 prob_h2: float = 0.15,
                 prob_h3: float = 0.1,
                 prob_h4: float = 0.0001,
                 n_node_features: int = 168,
                 n_edge_features: int = 2,
                 n_node_neurons: int = None,
                 n_edge_neurons: int = None,
                 n_gc_layers: int = 2,
                 n_s_layers: int = 4,
                 n_ts_layers: int = 2,
                 ):
        super(BuildNN_GNN_MTL, self).__init__()

        """
            A class that creates a model with the desired number of shared core
            layers (fully connected) + target specific core layers (fully connected)
            + Graph-Convolutional (n_gc_layers) + fully-connected (n_fc_layers)
            linear layers for GNN, using the specified non-linear activation layers
            interspaced between them.

            Args:
            :param int ni: number of input features; default: 4
            :param int n0: number of neurons in first layer of shared core; default: 200
            :param int n1: number of neurons in second layer of shared core; default: 100
            :param int n2: number of neurons in third layer of shared core; default: 50
            :param int n3: number of neurons in fourth layer of shared core; default: 10
            :param str act: activation function; can be any activation available in torch; default: relu
            :param float momentum_batch_norm: momentum for batch normalization; default: 0.9
            :param int prob_h1: dropout in first layer of shared core; default: 0.25
            :param int prob_h2: dropout in first layer of shared core; default: 0.15
            :param int prob_h3: dropout in first layer of shared core; default: 0.1
            :param int prob_h4: dropout in first layer of shared core; default: 0.0001
            :param int n_node_features: length of node feature vectors for GNN; default: 4
            :param int n_edge_features: length of edge feature vectors for GNN; default: 3
            :param int n_node_neurons: number of neurons in deep layers (if they exist)
                    in the densely-connected network if set to None, it is internally
                    set to n_node_features
            :param int n_edge_neurons: edges might have on input relatively few features (2,3);
                    this parameter allows the user to use a linear layer to expand the number
                    of edge features in the graph-convolution part.
            :param int n_gc_layers: number of Graph-Convolution layers (CGConv); default: 2
            :param int n_s_layers: number of layers in shared core; default: 4
            :param int n_ts_layers: number of layers in target specific core; default: 2

            """
        self.activation_layer = eval("nn." + act + "()")

        # GNN
        self.GNNlinear1 = nn.Linear(n_node_features, n_node_neurons) #if n_node_neurons > n_node_features else None
        self.GNNlinear2 = nn.Linear(n_edge_features, n_edge_neurons) #if n_edge_neurons > n_edge_features else None
        self.conv1 = CGConv(n_node_neurons, dim=n_edge_neurons)
        self.conv2 = CGConv(n_node_neurons, dim=n_edge_neurons)

        # Shared core
        self.linear1 = nn.Linear(n_node_neurons, n0)
        self.bn1 = nn.BatchNorm1d(n0, momentum=momentum_batch_norm)
        self.dropout1 = nn.Dropout(prob_h1)
        self.linear2 = nn.Linear(n0, n1)
        self.bn2 = nn.BatchNorm1d(n1, momentum=momentum_batch_norm)
        self.dropout2 = nn.Dropout(prob_h2)
        self.linear3 = nn.Linear(n1, n2)
        self.bn3 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.dropout3 = nn.Dropout(prob_h3)
        self.linear4 = nn.Linear(n2, n3)
        self.bn4 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.dropout4 = nn.Dropout(prob_h4)

        # Target specific core
        self.ts1_linear1 = nn.Linear(n3, n2)  # x.size(1)
        self.ts1_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts1_dropout1 = nn.Dropout(prob_h2)
        self.ts1_linear2 = nn.Linear(n2, n3)
        self.ts1_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts1_dropout2 = nn.Dropout(prob_h3)
        self.ts1_sig = nn.Linear(n3, 1)

        self.ts2_linear1 = nn.Linear(n3, n2)  # x.size(1)
        self.ts2_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts2_dropout1 = nn.Dropout(prob_h2)
        self.ts2_linear2 = nn.Linear(n2, n3)
        self.ts2_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts2_dropout2 = nn.Dropout(prob_h3)
        self.ts2_sig = nn.Linear(n3, 1)

        self.ts3_linear1 = nn.Linear(n3, n2)  # x.size(1)
        self.ts3_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts3_dropout1 = nn.Dropout(prob_h2)
        self.ts3_linear2 = nn.Linear(n2, n3)
        self.ts3_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts3_dropout2 = nn.Dropout(prob_h3)
        self.ts3_sig = nn.Linear(n3, 1)

        self.ts4_linear1 = nn.Linear(n3, n2)  # x.size(1)
        self.ts4_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts4_dropout1 = nn.Dropout(prob_h2)
        self.ts4_linear2 = nn.Linear(n2, n3)
        self.ts4_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts4_dropout2 = nn.Dropout(prob_h3)
        self.ts4_sig = nn.Linear(n3, 1)

        self.ts5_linear1 = nn.Linear(n3, n2)  # x.size(1)
        self.ts5_bn1 = nn.BatchNorm1d(n2, momentum=momentum_batch_norm)
        self.ts5_dropout1 = nn.Dropout(prob_h2)
        self.ts5_linear2 = nn.Linear(n2, n3)
        self.ts5_bn2 = nn.BatchNorm1d(n3, momentum=momentum_batch_norm)
        self.ts5_dropout2 = nn.Dropout(prob_h3)
        self.ts5_sig = nn.Linear(n3, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN
        #if self.GNNlinear1:
        #    x = self.GNNlinear1(x)
        #    x = torch.tanh(x)

        #if self.GNNlinear:
        #    x = self.GNNlinear2(x)
        #    x = torch.tanh(x)

        x = self.GNNlinear1(x)
        edge_attr = self.GNNlinear2(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = torch.tanh(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.tanh(x)

        # Pooling layer
        #x = global_mean_pool(x, batch)
        x = global_add_pool(x, batch)

        # Shared core
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activation_layer(self.bn1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.activation_layer(self.bn2(x))
        x = self.dropout3(x)
        x = self.linear3(x)
        x = self.activation_layer(self.bn3(x))
        x = self.dropout4(x)
        x = self.linear4(x)
        x = self.activation_layer(self.bn4(x))

        # Target specific core
        # Two layers
        # STRAIN 1
        y1 = self.ts1_dropout1(x)
        y1 = self.ts1_linear1(y1)
        y1 = self.activation_layer(self.ts1_bn1(y1))
        y1 = self.ts1_dropout2(y1)
        y1 = self.ts1_linear2(y1)
        y1 = self.activation_layer(self.ts1_bn2(y1))
        y1 = self.ts1_sig(y1)
        y1 = y1.sigmoid()

        y2 = self.ts2_dropout1(x)
        y2 = self.ts2_linear1(y2)
        y2 = self.activation_layer(self.ts2_bn1(y2))
        y2 = self.ts2_dropout2(y2)
        y2 = self.ts2_linear2(y2)
        y2 = self.activation_layer(self.ts2_bn2(y2))
        y2 = self.ts2_sig(y2)
        y2 = y2.sigmoid()

        y3 = self.ts3_dropout1(x)
        y3 = self.ts3_linear1(y3)
        y3 = self.activation_layer(self.ts3_bn1(y3))
        y3 = self.ts3_dropout2(y3)
        y3 = self.ts3_linear2(y3)
        y3 = self.activation_layer(self.ts3_bn2(y3))
        y3 = self.ts3_sig(y3)
        y3 = y3.sigmoid()

        y4 = self.ts4_dropout1(x)
        y4 = self.ts4_linear1(y4)
        y4 = self.activation_layer(self.ts4_bn1(y4))
        y4 = self.ts4_dropout2(y4)
        y4 = self.ts4_linear2(y4)
        y4 = self.activation_layer(self.ts4_bn2(y4))
        y4 = self.ts4_sig(y4)
        y4 = y4.sigmoid()

        y5 = self.ts5_dropout1(x)
        y5 = self.ts5_linear1(y5)
        y5 = self.activation_layer(self.ts5_bn1(y5))
        y5 = self.ts5_dropout2(y5)
        y5 = self.ts5_linear2(y5)
        y5 = self.activation_layer(self.ts5_bn2(y5))
        y5 = self.ts5_sig(y5)
        y5 = y5.sigmoid()

        return y1, y2, y3, y4, y5