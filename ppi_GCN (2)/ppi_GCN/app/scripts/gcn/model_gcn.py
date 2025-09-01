from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNLinkPredictor(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb = nn.Embedding(num_nodes, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Pairwise scoring: [h_u, h_v, |h_u-h_v|, h_u*h_v]
        pair_in = hidden_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(pair_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode(self, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
        # Build initial node features from a learned embedding table
        x0 = self.emb(torch.arange(self.num_nodes, device=device))
        h = self.conv1(x0, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

    def score_pairs(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: [B, 2], each row (u,v)
        u = h[pairs[:, 0]]
        v = h[pairs[:, 1]]
        feats = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        logits = self.mlp(feats).squeeze(-1)
        return logits

    def forward(self, edge_index: torch.Tensor, pairs: torch.Tensor, device: torch.device) -> torch.Tensor:
        h = self.encode(edge_index, device)
        return self.score_pairs(h, pairs)
