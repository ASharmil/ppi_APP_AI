import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from typing import List, Dict, Tuple

class DrugProteinMatcher(nn.Module):
    def __init__(
        self,
        drug_features: int = 200,  # RDKit molecular descriptors
        protein_embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.drug_features = drug_features
        self.protein_embed_dim = protein_embed_dim
        
        # Drug feature processor
        self.drug_embed = nn.Sequential(
            nn.Linear(drug_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Protein embedding processor
        self.protein_embed = nn.Sequential(
            nn.Linear(protein_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, drug_features: torch.Tensor, protein_embeddings: torch.Tensor):
        # Process drug features
        drug_embed = self.drug_embed(drug_features)  # [batch, hidden_dim]
        protein_embed = self.protein_embed(protein_embeddings)  # [batch, hidden_dim]
        
        # Add sequence dimension for attention
        drug_embed = drug_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        protein_embed = protein_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention between drug and protein
        attn_output, attn_weights = self.cross_attention(
            drug_embed, protein_embed, protein_embed
        )
        
        # Combine embeddings
        combined = torch.cat([
            drug_embed.squeeze(1),
            attn_output.squeeze(1)
        ], dim=1)
        
        # Predictions
        affinity_score = self.prediction_head(combined)
        confidence = self.confidence_head(combined)
        
        return {
            "affinity_score": affinity_score.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "attention_weights": attn_weights
        }

class DrugFeaturizer:
    """Convert SMILES to molecular descriptors using RDKit"""
    
    @staticmethod
    def smiles_to_features(smiles: str) -> np.ndarray:
        """Convert SMILES string to feature vector"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(200)  # Return zeros for invalid SMILES
            
            # Calculate RDKit descriptors
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.BertzCT(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                # Add more descriptors to reach ~200 features
            ]
            
            # Pad or truncate to exactly 200 features
            features = features[:200]
            while len(features) < 200:
                features.append(0.0)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return np.zeros(200)