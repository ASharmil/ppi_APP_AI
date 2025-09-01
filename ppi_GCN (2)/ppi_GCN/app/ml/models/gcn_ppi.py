import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Tuple, Optional

class PPIGraphNet(nn.Module):
    def __init__(
        self,
        node_features: int = 20,  # Amino acid features
        edge_features: int = 4,   # Contact/distance features
        haddock_features: int = 12,  # HADDOCK energy terms
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        heads: int = 4
    ):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.haddock_features = haddock_features
        self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embed = nn.Linear(node_features, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim if i > 0 else hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_features if edge_features > 0 else None
            )
            for i in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # HADDOCK feature processor
        self.haddock_processor = nn.Sequential(
            nn.Linear(haddock_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction heads
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # graph + haddock features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.binding_affinity_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Regression output
        )
        
        # Residue-level attention for interpretability
        self.residue_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_data: Batch, haddock_features: torch.Tensor):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        
        # Node embeddings
        x = self.node_embed(x)
        x = F.relu(x)
        
        # Graph convolutions with residual connections
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x + residual)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Pool to graph level
        graph_embed_mean = global_mean_pool(x, batch)
        graph_embed_max = global_max_pool(x, batch)
        graph_embed = torch.cat([graph_embed_mean, graph_embed_max], dim=1)
        
        # Process HADDOCK features
        haddock_embed = self.haddock_processor(haddock_features)
        
        # Combine features
        combined_features = torch.cat([graph_embed, haddock_embed], dim=1)
        
        # Predictions
        interaction_score = self.interaction_head(combined_features)
        binding_affinity = self.binding_affinity_head(combined_features)
        
        # Residue-level attention for interpretability
        node_embeddings = x.view(-1, 1, self.hidden_dim)  # Reshape for attention
        attn_weights, _ = self.residue_attention(
            node_embeddings, node_embeddings, node_embeddings
        )
        
        return {
            "interaction_score": interaction_score.squeeze(-1),
            "binding_affinity": binding_affinity.squeeze(-1),
            "residue_attention": attn_weights.squeeze(1),
            "node_embeddings": x
        }

class PPIDataBuilder:
    """Builds graph data from protein sequences and HADDOCK features"""
    
    # Standard amino acid properties (hydrophobicity, charge, size, etc.)
    AA_PROPERTIES = {
        'A': [1.8, 0, 0, 0, 0.62],   # Ala
        'R': [-4.5, 1, 1, 0, 0.64],  # Arg
        'N': [-3.5, 0, 1, 0, 0.60],  # Asn
        'D': [-3.5, -1, 1, 0, 0.55],  # Asp
        'C': [2.5, 0, 0, 0, 0.62],   # Cys
        'Q': [-3.5, 0, 1, 0, 0.68],  # Gln
        'E': [-3.5, -1, 1, 0, 0.68],  # Glu
        'G': [-0.4, 0, 0, 0, 0.48],  # Gly
        'H': [-3.2, 0, 1, 1, 0.72],  # His
        'I': [4.5, 0, 0, 0, 0.87],   # Ile
        'L': [3.8, 0, 0, 0, 0.87],   # Leu
        'K': [-3.9, 1, 1, 0, 0.82],  # Lys
        'M': [1.9, 0, 0, 0, 0.84],   # Met
        'F': [2.8, 0, 0, 1, 0.93],   # Phe
        'P': [-1.6, 0, 0, 0, 0.64],  # Pro
        'S': [-0.8, 0, 1, 0, 0.53],  # Ser
        'T': [-0.7, 0, 1, 0, 0.58],  # Thr
        'W': [-0.9, 0, 0, 1, 1.14],  # Trp
        'Y': [-1.3, 0, 1, 1, 1.06],  # Tyr
        'V': [4.2, 0, 0, 0, 0.76],   # Val
        'X': [0.0, 0, 0, 0, 0.64]    # Unknown
    }
    
    @classmethod
    def sequence_to_features(cls, sequence: str) -> np.ndarray:
        """Convert amino acid sequence to feature matrix"""
        features = []
        for i, aa in enumerate(sequence):
            aa_props = cls.AA_PROPERTIES.get(aa.upper(), cls.AA_PROPERTIES['X'])
            # Add positional encoding
            pos_enc = [
                np.sin(i / 10000),
                np.cos(i / 10000),
                i / len(sequence)  # Relative position
            ]
            features.append(aa_props + pos_enc)
        return np.array(features, dtype=np.float32)
    
    @classmethod
    def build_contact_edges(cls, sequence: str, contact_threshold: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges based on sequence adjacency and predicted contacts"""
        n = len(sequence)
        edge_index = []
        edge_attr = []
        
        # Sequence adjacency edges
        for i in range(n - 1):
            edge_index.extend([(i, i + 1), (i + 1, i)])
            edge_attr.extend([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])  # adjacent, not contact, not long-range, bond
        
        # Add some long-range contacts (simplified prediction)
        for i in range(n):
            for j in range(i + 3, min(i + 10, n)):
                if abs(i - j) > 2:  # Skip adjacent residues
                    # Simplified contact prediction based on amino acid properties
                    aa_i = cls.AA_PROPERTIES.get(sequence[i].upper(), cls.AA_PROPERTIES['X'])
                    aa_j = cls.AA_PROPERTIES.get(sequence[j].upper(), cls.AA_PROPERTIES['X'])
                    
                    # Simple heuristic for contact probability
                    contact_prob = 0.5 * (1 - abs(aa_i[0] - aa_j[0]) / 10)  # Hydrophobicity similarity
                    if contact_prob > 0.3:  # Threshold for adding edge
                        distance = abs(i - j) * 3.8  # Approximate CA-CA distance
                        edge_index.extend([(i, j), (j, i)])
                        edge_attr.extend([
                            [0.0, 1.0, 1.0, contact_prob],  # not adjacent, contact, long-range, strength
                            [0.0, 1.0, 1.0, contact_prob]
                        ])
        
        return np.array(edge_index).T, np.array(edge_attr)
    
    @classmethod
    def build_graph_data(cls, sequence_a: str, sequence_b: str, haddock_features: Optional[Dict] = None) -> Tuple[Data, torch.Tensor]:
        """Build PyTorch Geometric data object for protein pair"""
        
        # Node features
        features_a = cls.sequence_to_features(sequence_a)
        features_b = cls.sequence_to_features(sequence_b)
        
        # Build edges for each protein
        edge_index_a, edge_attr_a = cls.build_contact_edges(sequence_a)
        edge_index_b, edge_attr_b = cls.build_contact_edges(sequence_b)
        
        # Offset edges for protein B
        edge_index_b = edge_index_b + len(sequence_a)
        
        # Add inter-protein edges (simplified interface prediction)
        inter_edges = []
        inter_attrs = []
        for i in range(min(10, len(sequence_a))):  # Sample interface residues
            for j in range(min(10, len(sequence_b))):
                j_offset = j + len(sequence_a)
                inter_edges.extend([(i, j_offset), (j_offset, i)])
                inter_attrs.extend([[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5]])
        
        # Combine all features
        x = torch.tensor(np.vstack([features_a, features_b]), dtype=torch.float32)
        edge_index = torch.tensor(
            np.hstack([edge_index_a, edge_index_b, np.array(inter_edges).T]),
            dtype=torch.long
        )
        edge_attr = torch.tensor(
            np.vstack([edge_attr_a, edge_attr_b, inter_attrs]),
            dtype=torch.float32
        )
        
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # HADDOCK features tensor
        if haddock_features is None:
            # Default HADDOCK-style features
            haddock_tensor = torch.tensor([
                [-50.0, -30.0, -20.0, 15.0, 25.0, 5.0, 0.8, 0.6, 0.7, 1.2, 0.9, 2.1]
            ], dtype=torch.float32)
        else:
            haddock_tensor = torch.tensor([list(haddock_features.values())], dtype=torch.float32)
        
        return graph_data, haddock_tensor