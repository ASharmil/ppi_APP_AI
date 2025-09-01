from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Any

class DataSplitter:
    @staticmethod
    def create_splits(
        data: List[Any],
        labels: List[float],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List, List, List, List, List, List]:
        """Create train/val/test splits"""
        
        # First split: train+val vs test
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, 
            test_size=val_size_adjusted, 
            random_state=random_state,
            stratify=train_val_labels
        )
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels