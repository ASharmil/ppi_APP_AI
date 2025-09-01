import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from app.ml.models.drug_matcher import DrugProteinMatcher
from app.core.config import settings

class DrugTrainer:
    def __init__(
        self,
        model: DrugProteinMatcher,
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
        weight_decay: float = 1e-4
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
            patience=5,
            factor=0.5
        )
    
    def compute_loss(self, outputs: Dict, targets: torch.Tensor):
        """Compute loss for drug-protein matching"""
        affinity_loss = nn.BCELoss()(outputs['affinity_score'], targets.float())
        
        # Confidence regularization
        confidence_reg = torch.mean(torch.abs(outputs['confidence'] - outputs['affinity_score']))
        
        total_loss = affinity_loss + 0.1 * confidence_reg
        
        return {
            'total_loss': total_loss,
            'affinity_loss': affinity_loss,
            'confidence_reg': confidence_reg
        }
    
    def compute_metrics(self, outputs: Dict, targets: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        affinity_pred = outputs['affinity_score'].cpu().numpy()
        confidence_pred = outputs['confidence'].cpu().numpy()
        
        # ROC-AUC and Average Precision
        roc_auc = roc_auc_score(targets, affinity_pred)
        avg_precision = average_precision_score(targets, affinity_pred)
        
        # Confidence calibration
        confidence_diff = np.mean(np.abs(confidence_pred - affinity_pred))
        
        return {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confidence_calibration': confidence_diff
        }
    
    async def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        for drug_features, protein_embeddings, targets in train_loader:
            drug_features = drug_features.to(self.device)
            protein_embeddings = protein_embeddings.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(drug_features, protein_embeddings)
            losses = self.compute_loss(outputs, targets)
            
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            all_outputs.append(outputs)
            all_targets.extend(targets.cpu().numpy())
        
        # Compute epoch metrics
        combined_outputs = {
            'affinity_score': torch.cat([o['affinity_score'] for o in all_outputs]),
            'confidence': torch.cat([o['confidence'] for o in all_outputs])
        }
        
        metrics = self.compute_metrics(combined_outputs, np.array(all_targets))
        avg_loss = total_loss / len(train_loader)
        
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
            'val_losses': self.val_losses
        }
        
        checkpoint_path = Path(settings.MODELS_PATH) / "drug_matcher_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(settings.MODELS_PATH) / "drug_matcher_best.pt"
            torch.save(checkpoint, best_path)