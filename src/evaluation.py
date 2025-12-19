import torch
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from shared.models import EndOfTripDelay 
from shared.datasets import TripDataset
from data_filter import load_train_test_split


def evaluation(
    record_names:list[str],
    data_path:Path,
    exp_path:Path,
    config:dict,
):
    exp_dir = Path(exp_path)
    
    # 1. Load Scalers (Saved by TripTrainer.__init__)
    # We use weights_only=False because Scalers are often custom objects
    scalers = torch.load(exp_dir / "scalers.pth", map_location='cpu')
    std = scalers['delay'].std
    mu = scalers['delay'].mean

    # 2. Reconstruct Model Architecture
    model = EndOfTripDelay(
        in_dim=config['model']['in_dim'],
        gcn_dims=config['model']['gcn_dims'],  
        edge_attr_dim=config['model']['edge_attr_dim'],
        embd_dim=config['model']['embd_dim'],
        mlp_dims=config['model']['mlp_dims'],
        out_dim=config['model']['out_dim'],
    )

    checkpoint = torch.load(exp_dir / "checkpoints/best_model.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"--- Evaluating Experiment: {exp_dir.name} ---")

    all_data = []
    with torch.no_grad():
        for record in record_names:
            dataset = TripDataset(record, data_path, scalers)
            loader = DataLoader(dataset, batch_size=128)
            
            for data in loader:
                out = model(data)
                
                # We use mu here to get the actual "Clock Time" of the delay
                real_pred = (out * std) + mu
                real_actual = (data.y * std) + mu
                
                # Store for analysis
                batch_results = torch.stack([real_pred.flatten(), real_actual.flatten()], dim=1)
                all_data.append(batch_results.numpy())

    # 3. Final Interpretation
    results = np.vstack(all_data)
    errors = np.abs(results[:, 0] - results[:, 1])
    
    print(f"Final MAE: {np.mean(errors) / 60:.2f} minutes")
    
    # Save a CSV of predictions vs actuals for later visualization
    pd.DataFrame(results, columns=['pred_sec', 'actual_sec']).to_csv(exp_dir / "test_predictions.csv")


if __name__ == "__main__":

    # LAPTOP
    ROOT = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    
    # Docker env
    # ROOT = Path("/app")
    
    # PC 2080Ti
    # ROOT = Path("")

    CONFIG_PATH = ROOT / "src" / "config.yaml"
    DATA_DIR = ROOT / 'shared' / 'data'
    EXP_PATH = ROOT / 'shared' / 'experiments' / 'run_20251218_170803'

    # Load train config
    with open(CONFIG_PATH, 'r') as file:
        CONFIG = yaml.safe_load(file)
    
    # Load train/test splits
    RECORDS = load_train_test_split(DATA_DIR)
    TRAIN_DAYS = RECORDS['train'].copy()
    TEST_DAYS = RECORDS['test'].copy()

    # Run evaluation
    evaluation(
        record_names=TEST_DAYS,
        data_path=DATA_DIR,
        exp_path=EXP_PATH,
        config=CONFIG,
    )