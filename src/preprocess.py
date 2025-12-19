import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.data_filter import load_train_test_split, load_delays
from shared.scalers import StandardScaler, MinMaxScaler


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


if __name__ == "__main__":
    print("Saving scalers...")