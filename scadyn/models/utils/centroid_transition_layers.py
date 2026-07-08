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
class CentroidDynamicsPredictor(nn.Module):
    def __init__(self, 
                input_dim: int=256,
                hidden_dim: int=2048, 
                output_dim: int=256, 
                num_heads: int=4, 
                num_layers: int=1,
                num_centroids: int=25,
                temperature: float=1.0,
                apply_heads: bool=False,
                dropout: float = 0.1,
                fn: str='transformer',
                alpha: float=1.0,
                ema_decay: float=0.99,
                anneal_temp: bool=True,
                bias_in_last_fc: bool=True,
                centroid_update: str='default',
                target_transition: str='softmax',
                transition_type: str='patch',
                transition_logits_temp: float=1.0,
                input_transition: str='softmax',
                disable_layer_norm_before_fc: bool=True,
                patch_topk: int=3,
                loss_transition: dict=dict(name='wasserstein_loss', p=1, blur=0.01, weight=1.0),
                loss_futr_pred: dict=dict(type='SmoothL1Loss', loss_weight=1.0)):
        super().__init__()
        
        self.loss_transition = loss_transition
        if loss_transition['name'] == 'wasserstein_loss':
            self.loss_fn = SamplesLoss(loss='sinkhorn', p=loss_transition['p'], blur=loss_transition['blur'])
        elif loss_transition['name'] == 'wasserstein1_loss':
            self.loss_fn = SamplesLoss(loss='sinkhorn', p=loss_transition['p'], blur=loss_transition['blur'])

        self.with_futr_pred = loss_futr_pred is not None
        if self.with_futr_pred:
            self.loss_futr_pred = MODELS.build(loss_futr_pred)

        self.fname = fn
        self.alpha = alpha
        self.centroid_update = centroid_update
        self.num_centroids = num_centroids
        self.temperature = temperature
        self.target_transition = target_transition
        self.ema_decay = ema_decay
        self.transition_type = transition_type
        self.anneal_temp = anneal_temp
        self.num_heads = num_heads
        self.transition_logits_temp = transition_logits_temp
        self.input_transition = input_transition
        self.patch_topk = patch_topk
        self.disable_layer_norm_before_fc = disable_layer_norm_before_fc

        self.register_buffer('I', torch.eye(num_centroids))

        if not apply_heads:
            output_dim = num_centroids * 1
        else:
            output_dim = num_centroids * num_heads

        if centroid_update == 'learnable':
            self.projector = nn.Sequential(
                nn.Linear(input_dim*2, input_dim//4),
                nn.LayerNorm(input_dim//4),
                nn.ELU(),
                nn.Linear(input_dim//4, input_dim))
        
        if fn == 'transformer':
            self.fn = nn.Sequential(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=input_dim,
                        nhead=num_heads,
                        dim_feedforward=hidden_dim,
                        dropout=dropout),
                    num_layers=num_layers),
                nn.Linear(input_dim, output_dim)
            )
        elif fn == 'mlp':
            self.fn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), # not used in recent best results
                nn.ELU(),
                nn.Linear(hidden_dim, output_dim))
        elif fn == 'graph':
            self.graph = dgl.DGLGraph()
            self.graph.add_nodes(num_centroids)
            self.graph.add_edges(torch.arange(num_centroids), torch.arange(num_centroids))

            self.gat_layers = nn.ModuleList([
                dgl.nn.GATv2Conv(
                    in_feats = input_dim if i == 0 else hidden_dim * num_heads,
                    out_feats = hidden_dim,
                    num_heads = num_heads,
                    feat_drop = 0.0,
                    attn_drop = 0.0,
                    residual = True,
                    activation = F.elu if i < num_layers-1 else None,
                    allow_zero_in_degree = True
                ) for i in range(num_layers)])
            self.ln_layers = nn.ModuleList([
                nn.LayerNorm(hidden_dim*num_heads) for i in range(num_layers)])

            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim//2),
                nn.BatchNorm1d(hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, output_dim))

    def get_transition_matrix(self, 
                              t_cur_asm: Tensor,
                              t_futr_asm: Tensor,
                              t_cur_centroids: Tensor, 
                              t_futr_centroids: Tensor,
                              epoch: int,
                              patch_affinity: Tensor=None) -> Tensor:
        # transition matrix from teacher model as pseudo ground truth
        with torch.no_grad():
            if self.transition_type == 'affinity':
                # given the t_cur_asm and t_futr_asm, vote the patch affinity matrix to get the transition matrix
                B, N, K = t_cur_asm.size()
                device = patch_affinity.device
                
                # 1. Get cluster assignments
                cur_clusters = t_cur_asm.argmax(dim=-1)    # [B x THW]
                futr_clusters = t_futr_asm.argmax(dim=-1)  # [B x THW]
                
                # 2. Get top matches
                values, matched_indices = torch.topk(
                    patch_affinity, 
                    k=self.patch_topk, 
                    dim=2
                )  # [B x THW x k]
                
                # values = F.softmax(values, dim=2)  # [B x THW x k]
                
                # 3. Create batch indices
                batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, self.patch_topk)
                
                # 4. Get source and target cluster indices
                src_clusters = cur_clusters.unsqueeze(-1).expand(-1, -1, self.patch_topk)  # [B x THW x k]
                tgt_clusters = futr_clusters[batch_indices, matched_indices]  # [B x THW x k]
                
                # 5. Create sparse indices for all dimensions
                indices = torch.stack([
                    batch_indices.reshape(-1),    # batch dim
                    src_clusters.reshape(-1),     # source clusters
                    tgt_clusters.reshape(-1),     # target clusters
                ], dim=0)  # [3 x (B*THW*k)]
                
                # 6. Create transition matrix using sparse_coo_tensor
                values_flat = values.reshape(-1)  # [B*THW*k]
                transition = torch.sparse_coo_tensor(
                    indices,                      # [3 x (B*THW*k)]
                    values_flat,                  # [B*THW*k]
                    size=(B, K, K),
                    device=device).to_dense()
            elif self.transition_type == 'centroids':
                transition = F.cosine_similarity(t_cur_centroids.unsqueeze(2), t_futr_centroids.unsqueeze(1), dim=-1)

            tau = self.temperature
            if self.anneal_temp:
                tau = max(0.5, min(1.0, math.exp(max(-10, -0.001 * (epoch - 5))))) + 1e-6

            if self.target_transition == 'gumbel_softmax':
                transition = F.gumbel_softmax(transition, tau=tau, hard=False, dim=-1) # stochastic
            elif self.target_transition == 'softmax':
                transition = F.softmax(transition/tau, dim=-1)
            elif self.target_transition == 'default':
                transition = transition / (transition.sum(dim=-1, keepdim=True) + 1e-6)
            elif self.target_transition == 'normalize':
                transition = F.normalize(transition, p=1, dim=2)

        return transition
        
    def loss(self, 
             cur_centroids: Tensor,
             t_futr_asm: Tensor,
             t_futr_centroids: Tensor,  
             epoch: int=None,
             t_cur_centroids: Tensor=None,
             t_cur_asm: Tensor=None,
             patch_affinity: Tensor=None) -> Tuple[dict, Tensor, Tensor]:
        
        # predict transition matrix
        transition_matrix = self(cur_centroids)
        losses = dict()

        # get transition matrix from teacher model as pseudo ground truth
        t_delta = self.get_transition_matrix(
                        t_cur_asm,
                        t_futr_asm,
                        t_cur_centroids, 
                        t_futr_centroids, 
                        epoch,
                        patch_affinity)
        t_delta = t_delta.detach()
        
        if self.input_transition == 'softmax':
            transition_temp = self.transition_logits_temp
            delta_pred = F.softmax(transition_matrix/transition_temp, dim=-1)
        elif self.input_transition == 'gumbel_softmax':
            delta_pred = F.gumbel_softmax(transition_matrix, tau=self.transition_logits_temp, hard=False, dim=-1) # TODO: verify this

        # aux metrics
        with torch.no_grad():
            pred_idx = delta_pred.argmax(dim=-1)
            gt_idx = t_delta.argmax(dim=-1)
            losses['trans_acc'] = (pred_idx == gt_idx).float().mean()

        loss_transition_name = self.loss_transition['name']
        if loss_transition_name == 'wasserstein_loss':        
            loss = self.loss_fn(delta_pred, t_delta) #.mean() # NOTE: new
        elif loss_transition_name == 'cross_entropy_loss':
            loss = F.cross_entropy(transition_matrix.permute(0, 2, 1), gt_idx)
            # loss =  F.nll_loss(F.log_softmax(transition_matrix.permute(0, 2, 1), dim=-1), gt_idx)
        elif loss_transition_name == 'kl_div_loss':
            loss = F.kl_div(delta_pred.log(), t_delta, reduction='batchmean', log_target=False)
        elif loss_transition_name == 'l1_loss':
            loss = F.l1_loss(delta_pred, t_delta)
        elif loss_transition_name == 'mse_loss':
            loss = F.mse_loss(delta_pred, t_delta)
        elif loss_transition_name == 'smooth_l1_loss':
            loss = F.smooth_l1_loss(delta_pred, t_delta)

        if loss_transition_name != 'none':
            losses['loss_transition'] = loss * self.loss_transition['weight']

        # updated centroids using predicted transition matrix
        if self.centroid_update == 'residual':
            futr_centroids = (delta_pred + self.alpha * self.I) @ cur_centroids
        elif self.centroid_update == 'default':
            futr_centroids = delta_pred @ cur_centroids
        elif self.centroid_update == 'learnable':
            futr_centroids_default = delta_pred @ cur_centroids
            futr_centroids_default = torch.cat([futr_centroids_default, futr_centroids_default], dim=-1)
            futr_centroids = self.projector(futr_centroids_default)

        # aux metrics
        with torch.no_grad():
            losses['futr_sim'] = F.cosine_similarity(futr_centroids, t_futr_centroids.detach()).mean()
            losses['entropy'] = - torch.sum(delta_pred * torch.log(delta_pred + 1e-8), dim=-1).mean()

        #if self.loss_futr_pred is not None:
        if self.with_futr_pred:
            losses['loss_futr_pred'] = self.loss_futr_pred(futr_centroids, t_futr_centroids.detach())

        return losses, transition_matrix, futr_centroids

    def forward(self, centroids: Tensor) -> Tensor:
        # Predict transition matrix
        if self.fname == 'graph':
            b, n, d = centroids.shape
            centroids = centroids.flatten(0, 1)
            batched_graph = dgl.batch([self.graph] * b).to(centroids.device)

            for gat, norm in zip(self.gat_layers, self.ln_layers):
                hidden = gat(batched_graph, centroids)
                hidden = hidden.reshape(hidden.shape[0], -1)
                hidden = norm(hidden)
                centroids = hidden + centroids if centroids.shape == hidden.shape else hidden

            transition_matrix = self.fc(centroids)
            transition_matrix = rearrange(transition_matrix, '(b n) k -> b n k', b=b, n=n)
        else:
            transition_matrix = self.fn(centroids)

        return transition_matrix
    
    def predict(self, cur_centroids: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        transition_matrix = self(cur_centroids)

        if self.input_transition == 'softmax':
            delta_pred = F.softmax(transition_matrix/self.transition_logits_temp, dim=-1)
        else:
            delta_pred = transition_matrix
        
        # updated centroids using predicted transition matrix
        if self.centroid_update == 'residual':
            futr_centroids = (delta_pred + self.alpha * self.I) @ cur_centroids
        elif self.centroid_update == 'default':
            futr_centroids = delta_pred @ cur_centroids
        elif self.centroid_update == 'learnable':
            futr_centroids_default = delta_pred @ cur_centroids
            futr_centroids_default = torch.cat([futr_centroids_default, futr_centroids_default], dim=-1)
            futr_centroids = self.projector(futr_centroids_default)

        return transition_matrix, futr_centroids