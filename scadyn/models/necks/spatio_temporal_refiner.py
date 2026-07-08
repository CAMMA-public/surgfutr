from typing import Tuple, Union
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS
from geomloss import SamplesLoss
import dgl
from einops import rearrange

@MODELS.register_module()
class SpatioTemporalRefiner(nn.Module):
    def __init__(self, 
                    input_dim: int=384, 
                    hidden_dim: int=2048,
                    output_dim: int=384,
                    num_heads: int=8,
                    num_layers: int=2,
                    clip_len: int=8, 
                    num_patches: int=256, 
                    learnable_map_weights: bool=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.clip_len = clip_len
        self.num_patches = num_patches
        self.learnable_map_weights = learnable_map_weights

        self.spatio_temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim,  
                                        nhead=num_heads,
                                        dim_feedforward=hidden_dim,
                                        dropout=0.1,
                                        activation='gelu',
                                        batch_first=True),
            num_layers=num_layers)

        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.5))

        if not self.learnable_map_weights:
            self.w1.requires_grad = False
            self.w2.requires_grad = False

    def forward(self, 
            patch_features: Tensor, 
            attention_maps: Tensor):
        """
        Args:
            patch_features: [B, T, H, W, D]
            attention_maps: [B, num_heads, T, H*W, H*W]
        """

        # from attention_maps, create a block diagonal matrix of shape [B, num_heads, THW, THW]
        B, num_heads, T, HW, _ = attention_maps.shape
        THW = T * HW
        H, W = 16, 16
        
        block_diag = torch.zeros((B, THW, THW), dtype=attention_maps.dtype, device=attention_maps.device)
        for b in range(B):
            for t in range(T):
                block_diag[b, t * HW:(t + 1) * HW, t * HW:(t + 1) * HW] = attention_maps[b, 0, t].mean(dim=0) #Average across heads and assign

        # given the patch_features, apply a base transformer encoder to learn temporal dynamics
        # 1. flatten the patch_features
        patch_features = rearrange(patch_features, 'b t h w d -> b (t h w) d')

        # 2. apply the transformer encoder to learn temporal dynamics across multiple frames
        patch_features = self.spatio_temporal_encoder(patch_features)

        # 3. compute cosine similarity between patch_features
        patch_features = F.normalize(patch_features, p=2, dim=-1)
        similarity_matrix = torch.matmul(patch_features, patch_features.transpose(-2, -1))

        # 4. add similarity matrix to the block diagonal matrix using learnable map weights (if learnable_map_weights is True)
        similarity_matrix = similarity_matrix * self.w1 + block_diag * self.w2
        similarity_matrix = F.softmax(similarity_matrix, dim=-1) # or softmax?

        similarity_matrix = similarity_matrix.unsqueeze(1)
        patch_features = rearrange(patch_features, 'b (t h w) d -> b t h w d', t=T, h=H, w=W)

        return patch_features, similarity_matrix