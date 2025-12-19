import torch
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from shared.models import EndOfTripDelay 
from shared.datasets import TripDataset
from src.data_filter import load_train_test_split
from src.utils.time import name_to_day
from src.preprocess import load_scalers

import logging
logger = logging.getLogger("TripDelayGNN")

def evaluation(
    record_names:list[str],
    data_path:Path,
    exp_path:Path,
    config:dict,
):
    
    exp_dir = Path(exp_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Scalers (Saved by TripTrainer.__init__)
    # We use weights_only=False because Scalers are often custom objects
    scalers = load_scalers(data_path)
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
    logger.info(f"--- Loaded Model Checkpoint ---")
    checkpoint = torch.load(
        exp_dir / "checkpoints/best_model.pth", 
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'],)
    model.to(device)
    model.eval()

    logger.info(f"--- Evaluating Experiment: {exp_dir.name} ---")
    results = []
    with torch.no_grad():
        for record in record_names:
            dataset = TripDataset(record, data_path, scalers)
            loader = DataLoader(dataset, batch_size=1, shuffle=False) # 1 batch = 1 trip
            logger.info(f"\nEvaluating on Record: {record} | Total Trips: {len(dataset)}")
            for i,data in tqdm(enumerate(loader), desc=f"Evaluating {record}", total=len(loader)):
                data = data.to(device)
                
                out = model(data)
                
                # We use mu here to get the actual "Clock Time" of the delay
                real_pred = (out * std) + mu
                real_actual = (data.y * std) + mu
                
                # Store for analysis
                results.append({
                    'trip_id': dataset.trip_ids[i],
                    'bin_code': dataset.bin_codes[i],
                    'pred': real_pred.item(),
                    'actual': real_actual.item(),
                    'error': real_pred.item() - real_actual.item(),
                    'day': name_to_day(record)[0],
                })
        # End of record loop

    
    df = pd.DataFrame(results)
    df['abs_error'] = df['error'].abs()
    
    csv_path = exp_dir / "full_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\Evaluation Complete.")
    logger.info(f"CSV saved: {csv_path}")

    save_final_report(df, exp_dir)
    # visualize_daily_results(df, exp_dir) # LateX issue


def save_final_report(df: pd.DataFrame, exp_dir: Path):
    report_path = exp_dir / "final_eval_summary.txt"

    lines = []
    lines.append("=== GLOBAL GNN EVALUATION REPORT ===")
    lines.append(f"Total Test Samples: {len(df)}")
    lines.append(f"Global MAE:         {df['abs_error'].mean()} seconds")
    lines.append(f"Global Bias:        {df['error'].mean()} seconds")
    lines.append("")
    lines.append("--- PERFORMANCE BY DAY (MAE in seconds) ---")
    lines.append(df.groupby('day')['abs_error'].mean().to_string())
    lines.append("")
    lines.append("--- PERFORMANCE BY TIME BIN (MAE in seconds) ---")

    df = df.copy()
    df['clock_time'] = df['bin_code'].apply(
        lambda x: f"{int(x)//2:02d}:{(int(x)%2)*30:02d}"
    )
    lines.append(df.groupby('clock_time')['abs_error'].mean().to_string())

    report_text = "\n".join(lines)

    # write once
    report_path.write_text(report_text)

    # log the same content
    logger.info("\n" + report_text)

    logger.info(f"Summary report saved: {report_path}")


def visualize_daily_results(df: pd.DataFrame, exp_dir: Path):
    """
    Reads the global evaluation CSV and generates a sorted bar chart of MAE per day.
    """

    # 2. Calculate MAE per day in minutes
    # We use abs_error_sec / 60
    daily_mae = df.groupby('day')['abs_error'].mean()
    
    # Sort the days by MAE value for better visual comparison
    daily_mae = daily_mae.sort_values(ascending=True)

    # 3. Create the Visualization
    plt.figure(figsize=(12, 7))
    
    # Plot bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(daily_mae)))
    bars = plt.bar(daily_mae.index, daily_mae.values, color=colors, edgecolor='black', alpha=0.8)

    # Add LaTeX labels and title
    plt.title('GNN Performance: Mean Absolute Error MAE per Day', fontsize=15, pad=20)
    plt.xlabel('Evaluation Date', fontsize=12)
    plt.ylabel('MAE (seconds)', fontsize=12)

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f} s', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # 4. Save the plot
    plot_path = exp_dir / "daily_mae_performance.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Bar chart saved to: {plot_path}")


if __name__ == "__main__":

    # LAPTOP
    # ROOT = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    
    # Docker env
    # ROOT = Path("/app")
    
    # PC 2080Ti
    ROOT = Path("C:/Users/szeke/Desktop/vitmma19-pw-delay-detection")

    CONFIG_PATH = ROOT / "src" / "config.yaml"
    DATA_DIR = ROOT / 'shared' / 'data'
    EXP_PATH = ROOT / 'shared' / 'experiments' / 'run_20251219_104026'

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