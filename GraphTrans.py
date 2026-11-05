# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention import *
from edge_features import EdgeFeatures


class GatedCNNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)
        y = self.dropout(y)
        return self.norm(x + y)

class GatedCNN1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, kernel_size: int = 5, dropout: float = 0.1, num_layers: int = 3):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList([GatedCNNBlock(out_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj_in(x)
        for block in self.blocks: 
            y = block(y)
        return y


def significance_weighted_aggregation(h_V, mask):
 
    norm = torch.norm(h_V, dim=-1) 
    bool_mask = mask.bool() 
    norm_masked = norm.masked_fill(~bool_mask, float('-1e9')) 
    weights = F.softmax(norm_masked, dim=1) 
    weights = weights * mask 
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    H_agg = (weights.unsqueeze(-1) * h_V).sum(dim=1)
    return H_agg



class GraphTrans(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, num_encoder_layers=4, k_neighbors=30, augment_eps=0.0, dropout=0.2):
        super(GraphTrans, self).__init__()

        self.augment_eps = augment_eps

        self.EdgeFeatures = EdgeFeatures(edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.esm_dim = 2560
        self.cnn_out_dim = 256
        self.gated_cnn = GatedCNN1D(in_dim=self.esm_dim, out_dim=self.cnn_out_dim, kernel_size=5, dropout=dropout, num_layers=3)

        fused_dim = self.cnn_out_dim + (node_features - self.esm_dim) 

    
        self.W_v = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.drop_v = nn.Dropout(dropout)
        
       
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)

      
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        
        self.W_out = nn.Linear(hidden_dim * 2, 1, bias=True)

       
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, V, mask):
    
        E, E_idx = self.EdgeFeatures(X, mask)

        if self.training and self.augment_eps > 0:
            V = V + 0.1 * self.augment_eps * torch.randn_like(V)

        
        V_esm = V[:, :, :self.esm_dim]
        V_dssp = V[:, :, self.esm_dim:]

        
        V_cnn = self.gated_cnn(V_esm) 

        
        V_fused = torch.cat([V_cnn, V_dssp], dim=-1) 

    
        h_V = self.W_v(V_fused)
        h_V = self.norm_v(h_V)
        h_V = F.gelu(h_V) 
        h_V = self.drop_v(h_V)

        h_E = self.W_e(E)

    
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

      
        H_agg = significance_weighted_aggregation(h_V, mask) 
        H_agg_exp = H_agg.unsqueeze(1).expand(-1, h_V.size(1), -1) 
        h_V_fused = torch.cat([h_V, H_agg_exp], dim=-1) 

   
        logits = self.W_out(h_V_fused).squeeze(-1)
        return logits