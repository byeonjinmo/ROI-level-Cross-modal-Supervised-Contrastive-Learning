"""
Graph Neural Network Models for rs-fMRI Feature Extraction

Includes:
- GATClassifier: GAT-based classifier (standalone)
- GNNBackbone: Flexible GNN backbone supporting GCN, GraphSAGE, GAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool, global_max_pool


class GATClassifier(nn.Module):
    """Graph Attention Network for Depression Classification.

    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden layer dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_edge_attr: Whether to use edge weights
    """

    def __init__(self, in_channels=75, hidden_channels=64, num_heads=3,
                 dropout=0.3, use_edge_attr=True):
        super().__init__()
        self.use_edge_attr = use_edge_attr

        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=num_heads,
            dropout=dropout, edge_dim=1 if use_edge_attr else None
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels, heads=num_heads,
            dropout=dropout, edge_dim=1 if use_edge_attr else None
        )
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_channels * num_heads * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if self.use_edge_attr:
            x, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)

        if self.use_edge_attr:
            x, attn2 = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:
            x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)

        return x, [attn1, attn2]


class GNNBackbone(nn.Module):
    """GNN backbone supporting GCN, GraphSAGE, GAT.

    Args:
        in_dim: Input node feature dimension
        hidden: Hidden dimension
        model_type: 'gcn', 'sage', or 'gat'
        dropout: Dropout rate
        heads: Number of attention heads (GAT only)
    """

    def __init__(self, in_dim: int, hidden: int, model_type: str,
                 dropout: float = 0.25, heads: int = 4):
        super().__init__()
        model_type = model_type.lower()
        self.model_type = model_type

        if model_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif model_type == "sage":
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        elif model_type == "gat":
            self.conv1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
        else:
            raise ValueError("model_type must be one of: gcn, sage, gat")

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        g = global_mean_pool(x, data.batch)
        return g

    def forward_nodes(self, data: Data) -> torch.Tensor:
        """Return node embeddings before global pooling."""
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        return x

    def forward_nodes_batched(self, data: Data, num_nodes: int = 200) -> torch.Tensor:
        """Return node embeddings reshaped to (B, num_nodes, hidden_dim)."""
        node_emb = self.forward_nodes(data)
        batch_size = data.batch.max().item() + 1
        actual_nodes_per_graph = node_emb.shape[0] // batch_size
        assert actual_nodes_per_graph == num_nodes, \
            f"Expected {num_nodes} nodes per graph, got {actual_nodes_per_graph}"
        return node_emb.view(batch_size, num_nodes, -1)

    def forward_with_attention(self, data: Data):
        """Forward pass returning node embeddings AND attention weights (GAT only)."""
        if self.model_type != "gat":
            raise ValueError("forward_with_attention only works for GAT model")
        x, edge_index = data.x, data.edge_index
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.act(x)
        x = self.dropout(x)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, attn1, attn2

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_last(self):
        for param in self.conv2.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
