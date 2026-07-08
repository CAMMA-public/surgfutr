from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from einops import reduce, rearrange, repeat
from mmaction.utils import (OptConfigType, SampleList)

@MODELS.register_module()
class SinkhornClustering(BaseModule):
    def __init__(self,
                input_dim=384,
                output_dim=128,
                hidden_dim=2048,
                bottleneck_dim=256,
                K=10,
                hard_assign=False,
                marginals_temp_r=1.0,
                marginals_temp_c=1.0,
                sinkhorn_iterations=3,
                sinkhorn_lambda=1.0,
                uniform_marginals=False,
                enable_softmax_on_r=False,
                sum_heads=False,
                ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.K = K
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_lambda = sinkhorn_lambda
        self.uniform_marginals = uniform_marginals
        self.marginals_temp_r = marginals_temp_r
        self.marginals_temp_c = marginals_temp_c
        self.hard_assign = hard_assign
        self.enable_softmax_on_r = enable_softmax_on_r
        self.sum_heads = sum_heads

    def compute_r_marginals(self, attn_map, uniform_marginals, multiplier=1):
        # NOTE: for VideoMamba only, 
        if attn_map.ndim == 5:
            # compute cosine similarity between spatio-temporal features
            attn_map = rearrange(attn_map, 'b t h w c -> b (t h w) c')
            f_norm = F.normalize(attn_map, dim=-1, eps=1e-6) 
            r = torch.bmm(f_norm, f_norm.transpose(1, 2)) 

            # reduce the self loop similarity 
            r = r - 1000 * torch.eye(r.shape[1]).to(r.device)
            r = r.softmax(dim=-1).max(dim=1).values
            r = r / r.sum(dim=-1, keepdim=True)

            # check if nan values are present
            if torch.isnan(r).any():
                breakpoint()

            return r

        b, h, n, n = attn_map.shape
        # b, n, _ = attn_map.shape
        if uniform_marginals:
            n = multiplier * n
            r = (torch.ones([b, n]) / n).cuda()
        else:
            r = attn_map.mean(dim=1).max(dim=1).values
            r = r/r.sum(dim=-1, keepdim=True)
        return r
    
    def compute_c_marginals(self, r, indices, uniform_marginals, temp_r=1., temp_c=1.):
        b, n = r.shape
        k = indices.shape[1]
        if uniform_marginals:
            c = (torch.ones([b, 1, k]) / k).cuda()
            r = r[:, :, None]
        else:
            c = torch.einsum('b n, b k n -> b k', r, indices)
            c = rearrange(c / temp_c, 'b k -> b 1 k')
            c = F.softmax(c, dim=-1)
            r = rearrange(r, 'b n -> b n 1')
            r = F.softmax(r / temp_r, dim=1)
        return torch.nan_to_num(r), torch.nan_to_num(c)

    @torch.no_grad()
    def sinkhorn(self, cost, r, c):
        P = torch.exp(- self.sinkhorn_lambda * cost)
        P /= reduce(P, 'b n k -> b 1 1', reduction='sum')

        # Iterate over the sinkhorn algorithm
        for _ in range(self.sinkhorn_iterations):
            u = reduce(P, 'b n k -> b n 1', reduction='sum')
            P *= (r / u)
            u = reduce(P, 'b n k -> b 1 k', reduction='sum')
            P *= (c / u)

        P = torch.nan_to_num(P, nan=1e-8)
        return P, torch.sum(P * cost, dim=[1, 2])

    def forward(self, 
                feats: torch.Tensor, 
                attn_map: torch.Tensor,
                multiplier: int = 1) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # average norm of the centroids
        norm = torch.norm(feats[:, 0, :], dim=-1).mean()
        b, n, _ = feats.shape

        # compute row marginals
        r = self.compute_r_marginals(attn_map, self.uniform_marginals, multiplier)
        p = F.softmax(r/self.marginals_temp_r, dim=1)

        # sample indices (in integer till thw)
        try:
            indices = p.multinomial(num_samples=self.K, replacement=False)
        except RuntimeError as e:
            breakpoint()
            print(e)
            indices = p.multinomial(num_samples=self.K, replacement=True)

        indices = rearrange(indices, 'b k -> (b k)')
        indices = torch.eye(n).to(feats.device)[indices]
        indices = rearrange(indices, '(b k) n -> b k n', b=b)

        tokens_marginals, centroids_marginals = self.compute_c_marginals(
            r, indices, uniform_marginals=self.uniform_marginals, temp_r=self.marginals_temp_r,
            temp_c=self.marginals_temp_c
        ) # token_marginals: [b, n, 1], centroids_marginals: [b, 1, k] # NOTE: they sum to 1

        # initial centroids
        centroids = torch.einsum('b n d, b k n -> b k d', feats, indices) # b x k x d
        centroids = F.normalize(centroids, dim=-1) # normalize centroids is done , is it necessary to normalize before?
        assignments = torch.einsum('b n d, b k d -> b n k', feats, centroids) # NOTE: feats are normalized before this
        b, n, k = assignments.shape

        # compute costs
        cost = -assignments
        cost = (cost - cost.min()) / (cost.max() - cost.min() + 1e-6)

        # sinkhorn
        assignments, transportation_cost = self.sinkhorn(
            cost=cost,
            r=tokens_marginals,
            c=centroids_marginals)
        cost = transportation_cost

        # sum to 1 for each batch element
        normalized_assignments = assignments / assignments.sum(dim=1, keepdim=True) # b x n x k-

        if self.hard_assign:
            hard_assignments = torch.max(normalized_assignments, dim=-1, keepdim=True).values
            hard_assignments = repeat(hard_assignments, 'b n 1 -> b n k', k=k)
            hard_assignments = (normalized_assignments == hard_assignments).float()
            hard_assignments = normalized_assignments * hard_assignments
            
            # update centroids using hard assignments
            centroids = torch.einsum('b n d, b n k -> b k d', feats, hard_assignments)
        else:
            centroids = torch.einsum('b n d, b n k -> b k d', feats, normalized_assignments)

        return centroids, assignments