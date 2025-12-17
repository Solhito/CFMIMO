"""
train.py - Model training and validation
Includes data loading, training loop, validation, and model saving
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import json
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import seaborn as sns

from model import create_model, DeepUnfoldingGNN

warnings.filterwarnings('ignore')

class CellFreeDataset(Dataset):
    """Cell-Free Massive MIMO Dataset"""
    
    def __init__(self, data_file: str, device: torch.device = None):
        """
        Initialize dataset
        
        Args:
            data_file: Data file path
            device: Device
        """
        self.device = device if device else torch.device('cpu')
        
        # Load data
        with open(data_file, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.config = saved_data["config"]
        self.data = saved_data["data"]
        
        print(f"Dataset loaded: {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single sample"""
        sample = self.data[idx]
        
        # Convert to torch tensors
        R_all = torch.from_numpy(sample["R_all"]).to(self.device)
        
        # Add metadata
        result = {
            "R_all": R_all,
            "betas": torch.from_numpy(sample["betas"]).to(self.device),
            "ap_pos": torch.from_numpy(sample["ap_pos"]).to(self.device),
            "ue_pos": torch.from_numpy(sample["ue_pos"]).to(self.device),
            "eta": torch.from_numpy(sample["eta"]).to(self.device),
            "sigma2": torch.tensor(sample["sigma2"], device=self.device),
            "P_max": torch.tensor(sample["P_max"], device=self.device),
            "C_max": torch.tensor(sample["C_max"], device=self.device)
        }
        
        return result
    
    @property
    def system_config(self) -> Dict:
        """Get system configuration"""
        return self.config

class Trainer:
    """Model trainer"""
    
    def __init__(self, config: Dict, device: torch.device = None):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            device: Device
        """
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 10)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.grad_clip = config.get("grad_clip", 1.0)
        
        # Data paths
        self.data_dir = config.get("data_dir", "./data/cellfree_mimo")
        self.train_file = os.path.join(self.data_dir, "train_dataset.pkl")
        self.val_file = os.path.join(self.data_dir, "val_dataset.pkl")
        
        # Model saving path
        self.save_dir = config.get("save_dir", "./saved_models")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Results saving path
        self.results_dir = config.get("results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load system configuration
        sys_config_file = os.path.join(self.data_dir, "system_config.json")
        with open(sys_config_file, 'r') as f:
            self.system_config = json.load(f)
        
        # Update system configuration with training parameters
        self.system_config.update({
            "hidden_dim": config.get("hidden_dim", 64),
            "msg_passing_layers": config.get("msg_passing_layers", 3),
            "unfolding_layers": config.get("unfolding_layers", 5),
            "dropout": config.get("dropout", 0.1),
            "batch_size": config.get("batch_size", 10)
        })
        
        # Create model
        self.model = create_model(self.system_config, self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_se': [],
            'val_loss': [],
            'val_se': [],
            'lr': [],
            'train_power_violation': [],
            'train_fronthaul_violation': [],
            'val_power_violation': [],
            'val_fronthaul_violation': []
        }
        
        # Best model state
        self.best_model_state = None
        self.best_val_se = -float('inf')
        self.best_epoch = 0
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Gradient clipping: {self.grad_clip}")
        print(f"  Model save path: {self.save_dir}")
    
    def load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Load datasets"""
        print("\nLoading datasets...")
        
        # Training set
        train_dataset = CellFreeDataset(self.train_file, self.device)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Validation set
        val_dataset = CellFreeDataset(self.val_file, self.device)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, 8),  # Smaller batch size for validation
            shuffle=False,
            num_workers=0
        )
        
        print(f"  Training set: {len(train_dataset)} samples")
        print(f"  Validation set: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_se = 0
        total_power_violation = 0
        total_fronthaul_violation = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            R_all = batch["R_all"]  # (B, L, K, N, N)
            eta = batch["eta"]  # (B, K)
            
            
            # Forward pass
            outputs = self.model(R_all, eta)
            
            # Compute loss
            loss = self.model.compute_loss(outputs)
            
            # Get statistics
            batch_se = outputs['sum_se_final'].item()
            batch_power_violation = torch.mean(outputs['power_violation']).item()
            batch_fronthaul_violation = torch.mean(outputs['fronthaul_violation']).item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            # if self.grad_clip > 0:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Parameter update
            self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item()
            total_se += batch_se
            total_power_violation += batch_power_violation
            total_fronthaul_violation += batch_fronthaul_violation
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'SE': f'{batch_se:.4f}',
                'P_viol': f'{batch_power_violation:.4f}',
                'C_viol': f'{batch_fronthaul_violation:.4f}'
            })
            
            # Print detailed information every N batches
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{num_batches}: "
                      f"Loss={loss.item():.4f}, SE={batch_se:.4f}, "
                      f"Loss Components={self.model.loss_components}")
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_se = total_se / num_batches
        avg_power_violation = total_power_violation / num_batches
        avg_fronthaul_violation = total_fronthaul_violation / num_batches
        
        return avg_loss, avg_se, avg_power_violation, avg_fronthaul_violation
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_se = 0
        total_power_violation = 0
        total_fronthaul_violation = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Get data
                R_all = batch["R_all"]
                eta = batch["eta"]
                
                # Handle batch dimension for eta
                if len(eta.shape) == 1:
                    eta = eta.unsqueeze(0).expand(R_all.shape[0], -1)
                
                # Forward pass
                outputs = self.model(R_all, eta)
                
                # Compute loss
                loss = self.model.compute_loss(outputs)
                
                # Get statistics
                batch_se = outputs['sum_se_final'].item()
                batch_power_violation = outputs['power_violation'].item()
                batch_fronthaul_violation = outputs['fronthaul_violation'].item()
                
                # Accumulate statistics
                total_loss += loss.item()
                total_se += batch_se
                total_power_violation += batch_power_violation
                total_fronthaul_violation += batch_fronthaul_violation
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_se = total_se / num_batches
        avg_power_violation = total_power_violation / num_batches
        avg_fronthaul_violation = total_fronthaul_violation / num_batches
        
        return avg_loss, avg_se, avg_power_violation, avg_fronthaul_violation
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_se': self.best_val_se,
            'history': self.history,
            'config': self.config,
            'system_config': self.system_config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  Best model saved to: {best_path}")
    
    def save_history(self):
        """Save training history"""
        history_file = os.path.join(self.results_dir, 'training_history.pkl')
        
        with open(history_file, 'wb') as f:
            pickle.dump(self.history, f)
        
        # Also save as CSV for analysis
        csv_file = os.path.join(self.results_dir, 'training_history.csv')
        with open(csv_file, 'w') as f:
            f.write("epoch,train_loss,train_se,train_power_violation,train_fronthaul_violation,"
                   "val_loss,val_se,val_power_violation,val_fronthaul_violation,lr\n")
            for i in range(len(self.history['train_loss'])):
                f.write(f"{i},{self.history['train_loss'][i]},{self.history['train_se'][i]},"
                       f"{self.history['train_power_violation'][i]},{self.history['train_fronthaul_violation'][i]},"
                       f"{self.history['val_loss'][i]},{self.history['val_se'][i]},"
                       f"{self.history['val_power_violation'][i]},{self.history['val_fronthaul_violation'][i]},"
                       f"{self.history['lr'][i]}\n")
        
        print(f"Training history saved to: {history_file}")
    
    def plot_training_history(self):
        """Plot training history"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curve
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # SE curve
        axes[0, 1].plot(epochs, self.history['train_se'], 'b-', label='Train SE', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_se'], 'r-', label='Val SE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Sum SE (bit/s/Hz)', fontsize=12)
        axes[0, 1].set_title('Training and Validation SE', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[0, 2].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('Learning Rate', fontsize=12)
        axes[0, 2].set_title('Learning Rate Schedule', fontsize=14)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
        
        # Power constraint violation
        axes[1, 0].plot(epochs, self.history['train_power_violation'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_power_violation'], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Power Violation', fontsize=12)
        axes[1, 0].set_title('Power Constraint Violation', fontsize=14)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Fronthaul constraint violation
        axes[1, 1].plot(epochs, self.history['train_fronthaul_violation'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.history['val_fronthaul_violation'], 'r-', label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Fronthaul Violation', fontsize=12)
        axes[1, 1].set_title('Fronthaul Constraint Violation', fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Best validation SE marker
        axes[1, 2].axis('off')
        summary_text = (
            f"Training Summary\n"
            f"Best Epoch: {self.best_epoch}\n"
            f"Best Val SE: {self.best_val_se:.4f}\n"
            f"Final Train SE: {self.history['train_se'][-1]:.4f}\n"
            f"Final Val SE: {self.history['val_se'][-1]:.4f}\n\n"
            f"Config:\n"
            f"L: {self.system_config['L']}, K: {self.system_config['K']}\n"
            f"N: {self.system_config['N']}, τ_p: {self.system_config['tau_p']}\n"
            f"Unfolding Layers: {self.system_config['unfolding_layers']}"
        )
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'training_history.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {plot_file}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Deep Unfolding GNN model training")
        print("="*60)
        
        # Load datasets
        train_loader, val_loader = self.load_datasets()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epochs):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*40}")
            
            # Training
            train_loss, train_se, train_power_violation, train_fronthaul_violation = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_se'].append(train_se)
            self.history['train_power_violation'].append(train_power_violation)
            self.history['train_fronthaul_violation'].append(train_fronthaul_violation)
            
            # Validation
            val_loss, val_se, val_power_violation, val_fronthaul_violation = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_se'].append(val_se)
            self.history['val_power_violation'].append(val_power_violation)
            self.history['val_fronthaul_violation'].append(val_fronthaul_violation)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Print results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train SE: {train_se:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val SE: {val_se:.4f}")
            print(f"  Train Power Violation: {train_power_violation:.4f}, Train Fronthaul Violation: {train_fronthaul_violation:.4f}")
            print(f"  Val Power Violation: {val_power_violation:.4f}, Val Fronthaul Violation: {val_fronthaul_violation:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Check if best model
            is_best = False
            if val_se > self.best_val_se:
                self.best_val_se = val_se
                self.best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                is_best = True
                print(f"  🎉 New best model! Val SE: {val_se:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Save history and plot every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_history()
                self.plot_training_history()
            
            # Show estimated remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = avg_time_per_epoch * (self.epochs - epoch - 1)
            
            print(f"  Elapsed Time: {elapsed_time/60:.1f} minutes")
            print(f"  Estimated Remaining Time: {remaining_time/60:.1f} minutes")
        
        # Training completed
        total_time = time.time() - start_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model (epoch {self.best_epoch})")
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Best model at epoch {self.best_epoch}, Val SE: {self.best_val_se:.4f}")
        
        # Save final results
        self.save_history()
        self.plot_training_history()
        
        # Save training summary
        summary = {
            'total_time_minutes': total_time / 60,
            'best_epoch': self.best_epoch,
            'best_val_se': self.best_val_se,
            'final_train_se': self.history['train_se'][-1],
            'final_val_se': self.history['val_se'][-1],
            'final_train_power_violation': self.history['train_power_violation'][-1],
            'final_val_power_violation': self.history['val_power_violation'][-1],
            'final_train_fronthaul_violation': self.history['train_fronthaul_violation'][-1],
            'final_val_fronthaul_violation': self.history['val_fronthaul_violation'][-1],
            'config': self.config,
            'system_config': self.system_config,
            'history_keys': list(self.history.keys())
        }
        
        summary_file = os.path.join(self.results_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nTraining summary saved to: {summary_file}")
        print(f"Models saved to: {self.save_dir}")
        print(f"Results saved to: {self.results_dir}")

def main():
    """Main training function"""
    # Training configuration
    train_config = {
        # Data paths
        "data_dir": "./data/cellfree_mimo",
        "save_dir": "./saved_models",
        "results_dir": "./results",
        
        # Training parameters
        "epochs": 50,
        "batch_size": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        
        # Model parameters
        "hidden_dim": 64,
        "msg_passing_layers": 3,
        "unfolding_layers": 5,
        "dropout": 0.1
    }
    
    # Create trainer
    trainer = Trainer(train_config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Run training
    main()