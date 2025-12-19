import torch
import random
import time
import yaml
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_filter import load_train_test_split
from shared import EndOfTripDelay, TripDataset
from src.preprocess import load_scalers

import logging
logger = logging.getLogger("TripDelayGNN")


class TripTrainer:
    def __init__(self, model, scalers, device, config, base_log_dir="./experiments"):
        # 1. Create a unique folder for this run
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_log_dir) / run_id
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        # Configurations
        self.batch_size = config['training']['batch_size']
        self.lr = config['training']['lr']
        self.patience = config['training']['patience']

        # 2. Setup Logging
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "logs"))
        
        # 3. Save Scalers immediately (essential for later inference)
        torch.save(scalers, self.exp_dir / "scalers.pth")
        
        self.model = model.to(device)
        self.scalers = scalers
        self.device = device
        self.criterion = nn.HuberLoss(delta=1.0) # instead of nn.MSELoss(), to do not penalize outliers too much
        
         # 4. Optimizer and Scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=self.patience
        )
        
        self.best_loss = float('inf')
        self.global_step = 0


    def train_record(self, record_name, data_path):
        # Load dataset for this specific record
        dataset = TripDataset(record_name, data_path, self.scalers)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
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
            target = data.y
            
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
                logger.info(f"    Batch [{i+1}/{len(loader)}] | Real MAE: {batch_mae_real:.2f}")
                
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
            logger.info(f"--> Saved new best model to {self.exp_dir}")



def train(
    record_names:list[str], 
    data_path:Path, 
    exp_dir:Path,
    config:dict,
):
    
    logger.info(f"Starting training cycle with {len(record_names)} records.")
    logger.info(f"Tunable hyperparameters: Epochs={config['num_epochs']}, LR={config['training']['lr']}, Batch={config['training']['batch_size']}")
    
    # Load scalers
    scalers = load_scalers(data_path)

    # Init model
    model = EndOfTripDelay(
        in_dim=config['model']['in_dim'],
        gcn_dims=config['model']['gcn_dims'],  
        edge_attr_dim=config['model']['edge_attr_dim'],
        embd_dim=config['model']['embd_dim'],
        mlp_dims=config['model']['mlp_dims'],
        out_dim=config['model']['out_dim'],
    )

    logger.info("--- Model Architecture Summary ---")
    logger.info(str(model))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {trainable_params:,}")
    
    # Find
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init trainer
    trainer = TripTrainer(
        model, scalers, device, 
        base_log_dir=exp_dir, config=config
    )

    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        epoch_mse_values = []
        epoch_mae_values = []
        
        logger.info(f"\n>>> Epoch {epoch+1} / {num_epochs}")
        
        # Shuffle records so the model sees days in a different order each time
        random.shuffle(record_names)
        for record in record_names:
            logger.info(f"  Training on Record: {record}")
            
            # Run the training loop for this record
            avg_mse, avg_mae = trainer.train_record(record, data_path)
            
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
        
        logger.info(f"End of Epoch {epoch+1} | Avg Real MAE: {overall_epoch_mae:.2f}")
        
    return trainer.exp_dir


if __name__ == "__main__":
    
    # LAPTOP
    # ROOT = Path('/mnt/c/Users/rdsup/Desktop/vitmma19-pw-delay-detection')
    
    # Docker env
    # ROOT = Path("/app")
    
    # PC 2080Ti
    ROOT = Path("C:/Users/szeke/Desktop/vitmma19-pw-delay-detection")

    CONFIG_PATH = ROOT / "src" / "config.yaml"
    DATA_DIR = ROOT / 'shared' / 'data'
    EXP_DIR = ROOT / 'shared' / 'experiments'

    # Load train config
    with open(CONFIG_PATH, 'r') as file:
        CONFIG = yaml.safe_load(file)
    
    # Load train/test splits
    RECORDS = load_train_test_split(DATA_DIR)
    TRAIN_DAYS = RECORDS['train'].copy()
    TEST_DAYS = RECORDS['test'].copy()

    EXP_PATH = train(
        record_names=TRAIN_DAYS,
        data_path=DATA_DIR,
        exp_dir=EXP_DIR,
        config=CONFIG,
    )



