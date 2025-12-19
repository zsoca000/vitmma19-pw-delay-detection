
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool

class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1])) # batch norm
                layers.append(nn.LeakyReLU(0.01))
                layers.append(nn.Dropout(0.2)) # dropout
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GCNWithEdge(nn.Module):
    def __init__(self, layer_sizes: list[int], edge_attr_dim: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() # for batch norm

        for i in range(len(layer_sizes) - 1):
            nn_edge = nn.Sequential(
                nn.Linear(edge_attr_dim, layer_sizes[i]*layer_sizes[i+1]),
                nn.LeakyReLU(0.01),
                nn.Linear(layer_sizes[i]*layer_sizes[i+1], layer_sizes[i]*layer_sizes[i+1])
            )

            # Xavier init, avoid explosion
            torch.nn.init.xavier_uniform_(nn_edge[0].weight)
            torch.nn.init.xavier_uniform_(nn_edge[2].weight, gain=0.01) # Very small gain

            self.convs.append(NNConv(layer_sizes[i], layer_sizes[i+1], nn_edge, aggr='mean'))
            self.norms.append(nn.BatchNorm1d(layer_sizes[i+1])) # for batch norm

    def forward(self, x, edge_index, edge_attr):
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = norm(x) # Batch norm
            x = F.leaky_relu(x, negative_slope=0.01)
        x = self.convs[-1](x, edge_index, edge_attr)
        x = self.norms[-1](x) # Batch norm
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
            [embd_dim + 4] + mlp_dims + [out_dim]
        )

    def forward(self, data):
        
        # 1. GCN (runs on the "big graph" efficiently)
        x = self.GCN(data.x, data.edge_index, data.edge_attr)

        # 2. Filter nodes: only keep nodes that are part of the trips
        x_trips = x[data.trip_mask]
        # 3. Get the batch assignments for ONLY those trip nodes
        # This is crucial so pooling knows which nodes belong to which trip in the batch
        trip_batch = data.batch[data.trip_mask]
        # 4. Average trip pooling (per graph in the batch)
        # x will now have shape [batch_size, hidden_channels]
        x = global_mean_pool(x_trips, trip_batch, size=data.num_graphs)

        # 5. Meta embedding (day and sec are already [batch_size, 1])
        time_emb = data.time_emb # .view(-1, 1)
        x = torch.cat([x, time_emb], dim=-1)

        # 6. MLP for prediction
        return self.MLP(x)
