import torch
import torch.nn as nn
from torch_geometric.nn import Sequential
import torch.nn.functional as F

class BuildNN(nn.Module):
    def __init__(self,
                ni: int = 1348,
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

        """
            A class that creates a model with the desired number of shared core
            layers (fully connected) + target specific core layers (fully connected)
            + Graph-Convolutional (n_gc_layers) + fully-connected (n_fc_layers)
            linear layers for GNN, using the specified non-linear activation layers
            interspaced between them.

            Args:
            :param int ni: number of input features; default: 30
            :param int n0: number of neurons in first layer of shared core; default: 200
            :param int n1: number of neurons in second layer of shared core; default: 100
            :param int n2: number of neurons in third layer of shared core; default: 50
            :param int n3: number of neurons in fourth layer of shared core; default: 10
            :param str act: activation function; can be any activation available in torch; default: relu
            :param float momentum_batch_norm: momentum for batch normalization; default: 0.9
            :param int spec_layers: number of layers in target specific core; default: 2
            :param int prob_h1: dropout in first layer of shared core; default: 0.25
            :param int prob_h2: dropout in first layer of shared core; default: 0.15
            :param int prob_h3: dropout in first layer of shared core; default: 0.1
            :param int prob_h4: dropout in first layer of shared core; default: 0.0001
            

            :param int n_node_features: length of node feature vectors; default: 55
            :param int n_edge_features: length of edge feature vectors; currently 1,
                    the distance.
            :param int n_node_neurons: number of neurons in deep layers (if they exist)
                    in the densely-connected network if set to None, it is internally
                    set to n_node_features
            :param int n_edge_neurons: edges might have on input relatively few features (2,3);
                    this parameter allows the user to use a linear layer to expand the number
                    of edge features in the graph-convolution part.
            :param int n_gc_layers: number of Graph-Convolution layers (CGConv); default: 1
            :param int n_fc_layers: number of densely-connected layers (n_fc_layers); default: 1
            :param str activation: nonlinear activation layer; can be any activation
                    available in torch; default: Tanh
            :param str pooling: the type of pooling of node results; can be 'add',
                    i.e. summing over all nodes (e.g. to return total energy; default)
                    or 'mean', to return energy per atom.

            """
        self.activation_layer = eval("nn." + act + "()")
        # Shared core
        self.linear1 = nn.Linear(ni, n0)
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

    def forward(self, x):
        # Shared core
        #torch.manual_seed(42)  # Try resetting before dropout
        #print(torch.get_rng_state()[:5])  # Print first few elements
        x = self.dropout1(x)
        # Compute the dropout mask
        #dropout_mask = (x != 0).float()  # 1 for kept, 0 for dropped
        # Print or save mask
        #print("Dropout Mask:", dropout_mask)
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