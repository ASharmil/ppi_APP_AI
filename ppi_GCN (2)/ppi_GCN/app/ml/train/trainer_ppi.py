import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from loguru import logger
from app.ml.models.gcn_ppi import PPIGraphNet
from app.core.config import settings

class PPITrainer:
    def __init__(
        self,
        model: PPIGraphNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def setup_training(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10
    ):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=scheduler_patience,
            factor=0.5,
            verbose=True
        )
    
    def compute_loss(self, outputs: Dict, interaction_targets: torch.Tensor, affinity_targets: torch.Tensor):
        """Compute combined loss for interaction and binding affinity"""
        interaction_loss = nn.BCELoss()(outputs['interaction_score'], interaction_targets.float())
        affinity_loss = nn.MSELoss()(outputs['binding_affinity'], affinity_targets.float())
        
        # Weighted combination
        total_loss = 0.7 * interaction_loss + 0.3 * affinity_loss
        
        return {
            'total_loss': total_loss,
            'interaction_loss': interaction_loss,
            'affinity_loss': affinity_loss
        }
    
    def compute_metrics(self, outputs: Dict, interaction_targets: np.ndarray, affinity_targets: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        interaction_pred = outputs['interaction_score'].cpu().numpy()
        affinity_pred = outputs['binding_affinity'].cpu().numpy()
        
        # ROC-AUC for interaction prediction
        roc_auc = roc_auc_score(interaction_targets, interaction_pred)
        
        # PR-AUC for interaction prediction
        precision, recall, _ = precision_recall_curve(interaction_targets, interaction_pred)
        pr_auc = auc(recall, precision)
        
        # RMSE for binding affinity
        affinity_rmse = np.sqrt(np.mean((affinity_pred - affinity_targets) ** 2))
        
        # Pearson correlation for binding affinity
        affinity_corr = np.corrcoef(affinity_pred, affinity_targets)[0, 1]
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'affinity_rmse': affinity_rmse,
            'affinity_correlation': affinity_corr
        }
    
    async def train_epoch(self, train_loader: DataLoader, haddock_features: List[torch.Tensor]):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_interaction_targets = []
        all_affinity_targets = []
        
        for batch_idx, (batch_data, interaction_batch, affinity_batch) in enumerate(
            zip(train_loader, 
                torch.split(torch.cat([t for t in train_loader.dataset]), len(train_loader.dataset)),
                torch.split(torch.cat([t for t in train_loader.dataset]), len(train_loader.dataset)))
        ):
            batch_data = batch_data.to(self.device)
            batch_haddock = torch.stack(haddock_features[batch_idx * train_loader.batch_size:
                                                      (batch_idx + 1) * train_loader.batch_size]).to(self.device)
            
            interaction_targets = interaction_batch.to(self.device)
            affinity_targets = affinity_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_data, batch_haddock)
            losses = self.compute_loss(outputs, interaction_targets, affinity_targets)
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            all_outputs.append(outputs)
            all_interaction_targets.extend(interaction_targets.cpu().numpy())
            all_affinity_targets.extend(affinity_targets.cpu().numpy())
        
        # Compute epoch metrics
        combined_outputs = {
            'interaction_score': torch.cat([o['interaction_score'] for o in all_outputs]),
            'binding_affinity': torch.cat([o['binding_affinity'] for o in all_outputs])
        }
        
        metrics = self.compute_metrics(
            combined_outputs, 
            np.array(all_interaction_targets),
            np.array(all_affinity_targets)
        )
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, metrics
    
    async def validate_epoch(self, val_loader: DataLoader, haddock_features: List[torch.Tensor]):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_interaction_targets = []
        all_affinity_targets = []
        
        with torch.no_grad():
            for batch_idx, (batch_data, interaction_batch, affinity_batch) in enumerate(val_loader):
                batch_data = batch_data.to(self.device)
                batch_haddock = torch.stack(haddock_features[batch_idx * val_loader.batch_size:
                                                          (batch_idx + 1) * val_loader.batch_size]).to(self.device)
                
                interaction_targets = interaction_batch.to(self.device)
                affinity_targets = affinity_batch.to(self.device)
                
                outputs = self.model(batch_data, batch_haddock)
                losses = self.compute_loss(outputs, interaction_targets, affinity_targets)
                
                total_loss += losses['total_loss'].item()
                all_outputs.append(outputs)
                all_interaction_targets.extend(interaction_targets.cpu().numpy())
                all_affinity_targets.extend(affinity_targets.cpu().numpy())
        
        combined_outputs = {
            'interaction_score': torch.cat([o['interaction_score'] for o in all_outputs]),
            'binding_affinity': torch.cat([o['binding_affinity'] for o in all_outputs])
        }
        
        metrics = self.compute_metrics(
            combined_outputs,
            np.array(all_interaction_targets),
            np.array(all_affinity_targets)
        )
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(settings.MODELS_PATH) / "ppi_model_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(settings.MODELS_PATH) / "ppi_model_best.pt"
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # ROC-AUC
        train_roc = [m['roc_auc'] for m in self.train_metrics]
        val_roc = [m['roc_auc'] for m in self.val_metrics]
        axes[0, 1].plot(train_roc, label='Train ROC-AUC')
        axes[0, 1].plot(val_roc, label='Val ROC-AUC')
        axes[0, 1].set_title('ROC-AUC Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].legend()
        
        # PR-AUC
        train_pr = [m['pr_auc'] for m in self.train_metrics]
        val_pr = [m['pr_auc'] for m in self.val_metrics]
        axes[1, 0].plot(train_pr, label='Train PR-AUC')
        axes[1, 0].plot(val_pr, label='Val PR-AUC')
        axes[1, 0].set_title('PR-AUC Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('PR-AUC')
        axes[1, 0].legend()
        
        # Affinity RMSE
        train_rmse = [m['affinity_rmse'] for m in self.train_metrics]
        val_rmse = [m['affinity_rmse'] for m in self.val_metrics]
        axes[1, 1].plot(train_rmse, label='Train RMSE')
        axes[1, 1].plot(val_rmse, label='Val RMSE')
        axes[1, 1].set_title('Binding Affinity RMSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(settings.STORAGE_PATH) / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
