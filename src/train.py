import torch
import random
import time
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import global_mean_pool

from utils.time import name_to_day
from data_processing import (
    load_train_test_split,
    load_delays,
    load_trip_idx_list
)

# --------------- PREPROCESSING ---------------

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
                layers.append(nn.LeakyReLU(0.01))
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
                nn.LeakyReLU(0.01),
                nn.Linear(layer_sizes[i]*layer_sizes[i+1], layer_sizes[i]*layer_sizes[i+1])
            )

            # Xavier init, avoid explosion
            torch.nn.init.xavier_uniform_(nn_edge[0].weight)
            torch.nn.init.xavier_uniform_(nn_edge[2].weight, gain=0.01) # Very small gain

            self.convs.append(NNConv(layer_sizes[i], layer_sizes[i+1], nn_edge, aggr='mean'))

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.leaky_relu(x, negative_slope=0.01)
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
        x = global_mean_pool(x_trips, trip_batch)

        # 5. Meta embedding (day and sec are already [batch_size, 1])
        day = data.day.view(-1, 1) 
        sec = data.sec.view(-1, 1)
        x = torch.cat([x, day, sec], dim=-1)

        # 6. MLP for prediction
        return self.MLP(x)


# --------------- DATASET ---------------

class TripDataset(torch.utils.data.Dataset):
    def __init__(self, record_name, data_path:Path, scalers:dict):
        
        self.scalers = scalers
        self.record_name = record_name
        self.graphs_path = data_path / 'graphs'
        delays_path = data_path / 'delays'

        # Load graphs for the record
        self.load_graph_cache() # before filtering delays!!!
        
        # Nodes of the trips, by trip_id
        self.nodes_of_trips = load_trip_idx_list(self.graphs_path)
        
        # Which day?
        day = name_to_day(record_name)[0]
        
        # Load and filter (by available bins) delays
        df = load_delays(record_name,delays_path)
        df = self.filter_delays(df,dt=1/2)

        # Training data
        self.bin_codes = df['bin_code'].to_numpy()
        self.trip_ids = df.index.to_numpy()

        self.secs = scalers['sec'].transform(
            torch.tensor(df['trip_start'].to_numpy(),dtype=torch.float32).unsqueeze(-1)
        )
        
        day_tensor = torch.tensor([[day]], dtype=torch.float32)
        self.days = scalers['day'].transform(day_tensor) * torch.ones_like(self.secs)
        
        self.delays = scalers['delay'].transform(
            torch.tensor(df['delay'].to_numpy(),dtype=torch.float32).unsqueeze(-1)
        )
    
    def __getitem__(self, idx):
        
        g = self.g_cache[self.bin_codes[idx]]
        num_nodes = g.x.size(0)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[self.nodes_of_trips[self.trip_ids[idx]]] = True

        data = Data(
            x=g.x,
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            y=self.delays[idx],
            day=self.days[idx],
            sec=self.secs[idx],
            trip_mask=mask,
        )
        return data
    
    def __len__(self):
        return len(self.trip_ids)


    def load_graph_cache(self):
        self.g_cache = {}
        for f in (self.graphs_path / self.record_name).iterdir():
            if f.is_file() and f.name.startswith("graph_bin_") and f.suffix == ".pt":
                bin_code = int(f.stem.replace("graph_bin_", ""))
                g = torch.load(f, weights_only=False)
                g.x = self.scalers['x'].transform(g.x)
                g.edge_attr = self.scalers['edge_attr'].transform(g.edge_attr)
                self.g_cache[bin_code] = g
            

    def filter_delays(self, df:pd.DataFrame, dt=1/2):
        bins = np.arange(0,24+dt,dt)
        bins *= 60*60

        df['bin'] = pd.cut(df['trip_start'], bins=bins, right=False)
        df['bin_code'] = df['bin'].cat.codes
        return df[df['bin_code'].isin(self.g_cache.keys())]


# --------------- TRAINING LOGIC ---------------

class TripTrainer:
    def __init__(self, model, scalers, device, base_log_dir="./experiments"):
        # 1. Create a unique folder for this run
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_log_dir) / run_id
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        # 2. Setup Logging
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "logs"))
        
        # 3. Save Scalers immediately (essential for later inference)
        torch.save(scalers, self.exp_dir / "scalers.pth")
        
        self.model = model.to(device)
        self.scalers = scalers
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3
        )
        
        self.best_loss = float('inf')
        self.global_step = 0


    def train_record(self, record_name, data_path, epoch):
        # Load dataset for this specific record
        dataset = TripDataset(record_name, data_path, self.scalers)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get the standard deviation from the scaler (for delay)
        delay_std = self.scalers['delay'].std
        
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for i, data in enumerate(loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # 1. Forward pass
            out = self.model(data)
            target = data.y.view(-1, 1)
            
            # 2. Compute Loss (Z-Score space)
            loss = self.criterion(out, target)
            loss.backward()
            
            # 3. Advanced Strategy: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 4. Calculate Batch MAE in Real Units
            with torch.no_grad():
                batch_mae_real = torch.abs(out - target).mean().item() * delay_std
                
            # 5. LOGGING: Per Batch
            # Grouping by record name allows you to compare days in TensorBoard
            self.writer.add_scalar(f'Loss/Batch_MSE_Z/{record_name}', loss.item(), self.global_step)
            self.writer.add_scalar(f'Metric/Batch_MAE_Real/{record_name}', batch_mae_real, self.global_step)
            
            total_loss += loss.item()
            total_mae += batch_mae_real
            self.global_step += 1

            # Print every 50 batches
            if (i + 1) % 50 == 0:
                print(f"    Batch [{i+1}/{len(loader)}] | Real MAE: {batch_mae_real:.2f}")
                
        avg_mse = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        return avg_mse, avg_mae

    
    def save_checkpoint(self, current_loss, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_loss,
        }
        # Save "Last" model
        torch.save(checkpoint, self.exp_dir / "checkpoints" / "last_model.pth")
        
        # Save "Best" model
        if is_best:
            self.best_loss = current_loss
            torch.save(checkpoint, self.exp_dir / "checkpoints" / "best_model.pth")
            print(f"--> Saved new best model to {self.exp_dir}")



if __name__ == '__main__':

    num_epochs = 10

    # Path
    root = Path("/app/shared") # Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection/shared')
    data_path = root / 'data'
    experiments_path = root / 'experiments'
    
    # Load records dir (train/test)
    records = load_train_test_split(data_path)
    record_names = records['train'].copy()
    # Load scalers
    scalers = load_scalers(data_path)

    # Init model
    model = EndOfTripDelay(
        in_dim=7,
        gcn_dims=[32, 32],       # Increased for more capacity
        edge_attr_dim=2,
        embd_dim=16,
        mlp_dims=[64, 32],      # Wider MLP to prevent bottlenecks
        out_dim=1
    )
    
    # Find
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init trainer
    trainer = TripTrainer(model, scalers, device, base_log_dir=experiments_path)

    for epoch in range(num_epochs):
        epoch_mse_values = []
        epoch_mae_values = []
        
        print(f"\n>>> Epoch {epoch+1} / {num_epochs}")
        
        # Shuffle records so the model sees days in a different order each time
        random.shuffle(record_names)
        for record in record_names:
            print(f"  Training on Record: {record}")
            
            # Run the training loop for this record
            avg_mse, avg_mae = trainer.train_record(record, data_path, epoch)
            
            epoch_mse_values.append(avg_mse)
            epoch_mae_values.append(avg_mae)

        # Calculate Epoch-wide Averages
        overall_epoch_mse = np.mean(epoch_mse_values)
        overall_epoch_mae = np.mean(epoch_mae_values)
        
        # 6. LOGGING: Per Epoch
        trainer.writer.add_scalar('Epoch/Avg_MSE_Z', overall_epoch_mse, epoch)
        trainer.writer.add_scalar('Epoch/Avg_MAE_Real', overall_epoch_mae, epoch)
        
        # Update Scheduler & Checkpoint
        trainer.scheduler.step(overall_epoch_mse)
        
        is_best = overall_epoch_mse < trainer.best_loss
        trainer.save_checkpoint(overall_epoch_mse, epoch, is_best=is_best)
        
        print(f"End of Epoch {epoch+1} | Avg Real MAE: {overall_epoch_mae:.2f}")



    
    

    

