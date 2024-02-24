import numpy as np
from dgl.nn.pytorch import GraphConv, SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_metric_learning import losses

def gelu(x):
    """defines the GELU (Gaussian Error Linear Unit) activation function"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MultiHeadAttention(torch.nn.Module):
    """
    the multi-head attention module in PyTorch
    """
    # the initialization method for a multi-head attention module in PyTorch
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)


    def forward(self, X):
        """
        a forward pass function for a self-attention mechanism in a neural network
        """
        # (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


class EncoderLayer(torch.nn.Module):
    """
    ENCODER LAYER
    """
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        X = self.AN1(output + X)
        output = self.l1(X)
        X = self.AN2(output + X)
        return X


class feature_encoder(torch.nn.Module):
    """
    # multi-attention + the fully connected encoder layer
    """
    def __init__(self, vector_size, n_heads, n_layers, drop_out_rating):
        super(feature_encoder, self).__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)
        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)

        self.l3 = torch.nn.Linear(vector_size // 4, vector_size//2)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l4 = torch.nn.Linear(vector_size // 2, vector_size)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        X1 = self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)
        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)
        return X1, X2, X3, X5


class feature_encoder2(torch.nn.Module):
    """
    fully connected encoder layer
    """
    def __init__(self, vector_size, drop_out_rating):
        super(feature_encoder2, self).__init__()
        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)
        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.bn2 = torch.nn.BatchNorm1d(vector_size // 4)
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.dr(self.bn2(self.ac(self.l2(X))))
        return X

class graph_encoder(torch.nn.Module):
    """
    graph convolutional encoder on networks
    """
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, drug_num=0):
        super(graph_encoder, self).__init__()
        self.in_dim = in_dim
        self.drug_num = drug_num
        self.feat = nn.Parameter(torch.Tensor(in_dim, in_dim))
        nn.init.xavier_uniform_(self.feat, gain=nn.init.calculate_gain('relu'))
        self.conv1 = SAGEConv(in_dim, hidden1_dim, 'gcn')
        self.conv2 = SAGEConv(hidden1_dim, hidden2_dim, 'gcn')

    def forward(self, g, name=None):
        if name == 'drug_disease':
            # unweighted
            h = self.conv1(g, self.feat)
            h = F.relu(h)
            z = self.conv2(g, h)
            z_dr = z[:self.drug_num]
            z_ds = z[self.drug_num:]
            adj_rec = torch.sigmoid(torch.matmul(z_dr, z_ds.t()))
            return z_dr, z_ds, adj_rec
        else:
            # weighted
            h = self.conv1(g, self.feat, edge_weight=g.edata['weight'])
            h = F.relu(h)
            z = self.conv2(g, h, edge_weight=g.edata['weight'])
            adj_rec = torch.matmul(z, z.t())
            return z, adj_rec


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self):
        super(SupervisedContrastiveLoss, self).__init__()

    def forward(self, feature_vectors, labels): # CFV, target
        return losses.ContrastiveLoss()(feature_vectors, labels)


class Model(torch.nn.Module):
    """
    the full architecture of the model DrIGCL
    """
    def __init__(self, input_dr, input_ds, input_dr_ds, n_heads, n_layers, hidden1, hidden2, drop_out_rating, mp):
        super(Model, self).__init__()
        self.mp = mp
        self.input_dr = input_dr  # dimension of the drug feature
        self.input_ds = input_ds
        self.input_dr_ds = input_dr_ds
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.graphEncoder_output_dim = self.hidden2   # drug disease feature dim
        self.drop_out_rating = drop_out_rating
        # graph encoder
        self.drugEncoder = graph_encoder(self.input_dr, self.hidden1, self.hidden2)
        self.diseaseEncoder = graph_encoder(self.input_ds, self.hidden1, self.hidden2)
        self.drdsEncoder = graph_encoder(self.input_dr_ds, self.hidden1, self.hidden2, drug_num=input_dr)
        # drug latent feature fusion: 2 layers
        self.feaEncoder1_input_dim = self.graphEncoder_output_dim * 2  # Dr1,Ds2 dim
        self.feaEncoder2_input_dim = self.graphEncoder_output_dim * 2  # Dr2,Ds1 dim

        self.feaEncoder1 = feature_encoder2(self.feaEncoder1_input_dim, drop_out_rating)  # Dr1,Ds2 fusion
        self.feaEncoder2 = feature_encoder2(self.feaEncoder2_input_dim, drop_out_rating)  # Dr2,Ds1 fusion

        self.feaEncoder1_output_dim = self.feaEncoder1_input_dim // 4  # FD1
        self.feaEncoder2_output_dim = self.feaEncoder2_input_dim // 4  # FD2
        # drug feature fusion
        self.feaFui_input_dim = self.feaEncoder1_output_dim + self.feaEncoder2_output_dim + self.graphEncoder_output_dim *2
        # multi-head and auto-encoder
        self.feaFui = feature_encoder(self.feaFui_input_dim, n_heads, n_layers, drop_out_rating)
        self.linear_input_dim = self.feaFui_input_dim // 4 + self.feaFui_input_dim
        self.l1 = torch.nn.Linear(self.linear_input_dim, (self.linear_input_dim // 2))
        self.bn1 = torch.nn.BatchNorm1d(self.linear_input_dim // 2)
        self.l2 = torch.nn.Linear(self.linear_input_dim // 2, 1)
        self.ac = gelu
        self.dr = torch.nn.Dropout(drop_out_rating)
        self.criteria1 = torch.nn.BCEWithLogitsLoss()
        self.criteria3b = losses.TripletMarginLoss(triplets_per_anchor=1)


    def forward(self, graph_dr, graph_ds, train_graph, edges_ori):

        DR1, rec_dr_sim = self.drugEncoder(graph_dr)
        DS1, rec_ds_sim = self.diseaseEncoder(graph_ds)
        DR2, DS2, rec_drds = self.drdsEncoder(train_graph, name='drug_disease')

        drug_id, dis_id = edges_ori.T

        DR1 = DR1[drug_id]
        DS1 = DS1[dis_id]
        DR2 = DR2[drug_id]
        DS2 = DS2[dis_id]

        X1 = torch.cat((DR1, DS2), 1)
        X2 = torch.cat((DR2, DS1), 1)

        FD1 = self.feaEncoder1(X1)
        FD2 = self.feaEncoder2(X2)
        # CFV
        XC = torch.cat((DR2, FD1, FD2, DS2), 1)
        _, _, CFV, _ = self.feaFui(XC)  # multi-head att

        X = torch.cat((DR2, FD1, FD2, DS2, CFV), 1)
        # predict classification
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.l2(X)
        return X, CFV, rec_dr_sim, rec_ds_sim, rec_drds

    def get_loss(self, X, target, CFV,  loss_weight, cl_weight):
        """
        all loss function: classification + contrastive losses
        :param X: predicted labels
        :param target: the true label
        :param CFV: the embeddings of drug-disease pairs, used to calculate the contrastive loss
        :param loss_weight: weight of contrastive loss
        :param cl_weight: weight of classification loss
        :return: the total loss
        """
        loss_classification = self.criteria1(X.view(-1), target.view(-1))
        loss_contrastive = self.criteria3b(CFV, target)
        return cl_weight * loss_classification + loss_weight * loss_contrastive



