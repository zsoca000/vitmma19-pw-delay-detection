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
from torch.utils.data import DataLoader
from utils.time import name_to_day
from data_processing import (
    load_train_test_split,
    load_graph,
    load_delays,
    load_stop_id_to_idx,
    load_trip_idx_list
)

# --------------- PREPROCESSIN ---------------

class StandardScaler:
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        return self

    def transform(self, x: torch.Tensor):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: torch.Tensor):
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"]

class MinMaxScaler:
    def __init__(self, feature_range=(-1.0, 1.0), eps=1e-8):
        self.min = None
        self.max = None
        self.low, self.high = feature_range
        self.eps = eps

    def fit(self, x: torch.Tensor):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values
        return self

    def transform(self, x: torch.Tensor):
        x_std = (x - self.min) / (self.max - self.min + self.eps)
        return x_std * (self.high - self.low) + self.low

    def inverse_transform(self, x: torch.Tensor):
        x_std = (x - self.low) / (self.high - self.low)
        return x_std * (self.max - self.min + self.eps) + self.min

    def state_dict(self):
        return {
            "min": self.min,
            "max": self.max,
            "feature_range": (self.low, self.high),
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.min = state["min"]
        self.max = state["max"]
        self.low, self.high = state["feature_range"]
        self.eps = state["eps"]

def load_train_data(data_path:Path):
    # Graph and delay paths
    graphs_path = data_path / 'graphs'
    delays_path = data_path / 'delays'

    # Load training
    graphs_train, delays_train = [],[]
    records = load_train_test_split(data_path)
    for record_name in tqdm(records['train']):
        for load_path in (graphs_path / record_name).iterdir():
            graphs_train.append(torch.load(load_path, weights_only=False))
            
        delays_train.append(load_delays(record_name,delays_path)['delay'].to_numpy())

    # Train node and edge features
    x_train = torch.cat([g.x for g in graphs_train],axis=0)
    edge_attr_train =torch.cat([g.edge_attr for g in graphs_train],axis=0)

    # Train delays
    delays_train = torch.from_numpy(np.concatenate(delays_train, axis=0, dtype=np.float32))

    return x_train, edge_attr_train, delays_train

def save_scalers(data_path:Path):

    # Load training data      
    (
        x_train, 
        edge_attr_train, 
        delays_train
    ) = load_train_data(data_path)
    
    # Init scalers
    x_scaler = StandardScaler()
    edge_attr_scaler = StandardScaler()
    delay_scaler = StandardScaler()
    day_scaler = MinMaxScaler()
    sec_scaler = MinMaxScaler()

    # Fit scalers
    x_scaler.fit(x_train)
    edge_attr_scaler.fit(edge_attr_train)
    delay_scaler.fit(delays_train)
    day_scaler.fit(torch.tensor([0,6]))
    sec_scaler.fit(torch.tensor([0,24*60*60]))

    # Save scalers
    torch.save(
        {
            "x": x_scaler.state_dict(),
            "edge_attr": edge_attr_scaler.state_dict(),
            "delay": delay_scaler.state_dict(),
            "day": day_scaler.state_dict(),
            "sec": sec_scaler.state_dict(),
        },
        data_path / "scalers.pt"
)

def load_scalers(data_path:Path)->dict[MinMaxScaler|StandardScaler]:
    # Load checkpoint
    ckpt = torch.load(data_path / "scalers.pt", map_location="cpu")

    # Init empty scalers
    x_scaler = StandardScaler()
    edge_attr_scaler = StandardScaler()
    delay_scaler = StandardScaler()
    day_scaler = MinMaxScaler()
    sec_scaler = MinMaxScaler()

    # Load states
    x_scaler.load_state_dict(ckpt["x"])
    edge_attr_scaler.load_state_dict(ckpt["edge_attr"])
    delay_scaler.load_state_dict(ckpt["delay"])
    day_scaler.load_state_dict(ckpt["day"])
    sec_scaler.load_state_dict(ckpt["sec"])

    # Return scalers
    return {
        'x': x_scaler,
        'edge_attr' : edge_attr_scaler,
        "delay" : delay_scaler,
        "day" : day_scaler,
        "sec": sec_scaler,
    }

# --------------- MODELS ---------------

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


    
    


    
    

    

