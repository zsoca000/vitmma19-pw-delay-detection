import torch
import joblib
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from utils.time import name_to_day
from data_processing import (
    load_train_test_split,
    load_graph,
    load_delays,
    load_stop_id_to_idx,
    load_trip_idx_list
)


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GCNWithEdge(nn.Module):
    def __init__(self, layer_sizes: list[int], edge_attr_dim: int):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            nn_edge = nn.Sequential(
                nn.Linear(edge_attr_dim, layer_sizes[i]*layer_sizes[i+1]),
                nn.ReLU(),
                nn.Linear(layer_sizes[i]*layer_sizes[i+1], layer_sizes[i]*layer_sizes[i+1])
            )
            self.convs.append(NNConv(layer_sizes[i], layer_sizes[i+1], nn_edge, aggr='mean'))

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_attr))
        x = self.convs[-1](x, edge_index, edge_attr)
        return x
    

class EndOfTripDelay(nn.Module):
    def __init__(self, 
            in_dim: int, 
            gcn_dims:list[int], edge_attr_dim: int, 
            embd_dim: int,
            mlp_dims: list[int],
            out_dim: int,
        ):
        super().__init__()
        
        self.GCN = GCNWithEdge(
            [in_dim] + gcn_dims + [embd_dim], 
            edge_attr_dim
        )
        
        self.MLP = MLP(
            [embd_dim + 2] + mlp_dims + [out_dim]
        )

    def forward(self, x, edge_index, edge_attr, trip_nodes, day, sec):

        # GCN
        x = self.GCN(x, edge_index, edge_attr)

        # Average trip pooling
        x = x[trip_nodes].mean(dim=0)

        # Meta embedding
        x = torch.cat([x, day, sec])

        # MLP for prediction
        x = self.MLP(x)
        return x


    
    
    
    

    

