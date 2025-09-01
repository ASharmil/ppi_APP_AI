import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
from pathlib import Path
from app.core.config import settings

class MetricsCalculator:
    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
        """Generate and save confusion matrix plot"""
        y_pred_binary = (y_pred > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = Path(settings.STORAGE_PATH) / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    @staticmethod
    def generate_roc_curve(y_true: np.ndarray, y_pred: np.ndarray):
        """Generate ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        
        plot_path = Path(settings.STORAGE_PATH) / "roc_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    @staticmethod
    def generate_pr_curve(y_true: np.ndarray, y_pred: np.ndarray):
        """Generate Precision-Recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        plot_path = Path(settings.STORAGE_PATH) / "pr_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)