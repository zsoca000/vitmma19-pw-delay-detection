import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv  # or GraphSAGE, GAT, etc.

class GraphLSTM(nn.Module):
    def __init__(self, in_dim, gnn_hidden, lstm_hidden, out_dim):
        super().__init__()
        self.gnn = GCNConv(in_dim, gnn_hidden)
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTMCell(gnn_hidden, lstm_hidden)
        self.output_layer = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x, edge_index, h=None, c=None):
        """
        x: [num_nodes, in_dim] node features at current timestep
        edge_index: PyG edge index
        h, c: [num_nodes, lstm_hidden] optional previous hidden/cell states
        """
        if h is None:
            h = torch.zeros(x.size(0), self.lstm_hidden, device=x.device)
            c = torch.zeros_like(h)

        # GNN step
        gnn_out = torch.relu(self.gnn(x, edge_index))  # [num_nodes, gnn_hidden]

        # LSTM step
        h, c = self.lstm(gnn_out, (h, c))  # temporal update

        # Output prediction
        out = self.output_layer(h)  # [num_nodes, out_dim]
        return out, h, c
