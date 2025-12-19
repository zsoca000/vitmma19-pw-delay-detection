import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from utils.time import name_to_day
from data_filter import load_delays,load_trip_idx_list


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

        # The bins and the available trips
        self.bin_codes = df['bin_code'].to_numpy()
        self.trip_ids = df.index.to_numpy()

        # ADD Cyclical Encoding (instead of MinMaxScaler)
        # 1. Seconds (Period = 24*60*60)
        raw_secs = torch.tensor(df['trip_start'].to_numpy(), dtype=torch.float32).unsqueeze(-1)
        sec_norm = 2 * np.pi * raw_secs / (24*60*60)
        self.sec_sin = torch.sin(sec_norm)
        self.sec_cos = torch.cos(sec_norm)

        # 2. Days (Period = 7)
        raw_days = torch.tensor([[day]], dtype=torch.float32).expand_as(raw_secs)
        day_norm = 2 * np.pi * raw_days / 7
        self.day_sin = torch.sin(day_norm)
        self.day_cos = torch.cos(day_norm)
        
        # Delays
        self.delays = scalers['delay'].transform(
            torch.tensor(df['delay'].to_numpy(),dtype=torch.float32).unsqueeze(-1)
        )
    
    def __getitem__(self, idx):
        
        g = self.g_cache[self.bin_codes[idx]]
        num_nodes = g.x.size(0)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[self.nodes_of_trips[self.trip_ids[idx]]] = True

        time_emb=torch.cat([
            self.sec_sin[idx], self.sec_cos[idx], 
            self.day_sin[idx], self.day_cos[idx]
        ], dim=-1)

        # Shape [1, 4], for PyG batching
        time_emb = time_emb.unsqueeze(0)  
        
        # Shape [1,1], for PyG batching
        y = self.delays[idx].view(-1, 1)

        data = Data(
            x=g.x,
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            y=y,
            time_emb=time_emb,
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