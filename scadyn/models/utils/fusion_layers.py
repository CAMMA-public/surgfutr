from typing import Tuple, Union
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS
from geomloss import SamplesLoss
from einops import rearrange


@MODELS.register_module()
class FusionCentroidsWithFutrPred(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 feature_dim: int = 384,
                 num_clusters: int = 25,
                 num_heads: int = 4,
                 clip_len: int = 4):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.project_futr_pred = nn.Sequential(
            nn.Linear(self.feature_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim))

        new_seq_len = num_heads * num_clusters + clip_len
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.input_dim))

    def forward(self,
                futr_pred: Tensor,
                centroids: Tensor) -> Tensor:

        B, t, _ = futr_pred.shape
        B, hk, d2 = centroids.shape
        
        # Project futr_pred to match centroids dim
        projected_futr_pred = self.project_futr_pred(futr_pred)  # B x t x d2
        
        # Reshape and combine tokens
        projected_futr_pred = projected_futr_pred.reshape(B, t, 1, d2)
        centroids = centroids.reshape(B, 1, hk, d2)
        
        # Broadcast and concatenate
        combined = torch.cat([
            projected_futr_pred.expand(-1, -1, hk, -1),
            centroids.expand(-1, t, -1, -1)
        ], dim=-1)  # B x t x hk x (d2*2)
        
        # Project to final dimension
        output = self.fusion_mlp(combined)  # B x t x hk x output_dim

        # average over time
        output = output.mean(dim=1)

        return output
