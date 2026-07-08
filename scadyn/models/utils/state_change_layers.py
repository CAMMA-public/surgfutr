from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class StateChangeDecoder(BaseModule):
    """ State Change Decoder to predict class embeddings from the latent action embeddings."""
    _num_classes = {'verb': 10, 'triplet': 100, 'instrument-target': 59, 'instrument': 6, 'instrument-verb': 26}

    def __init__(self,
                 input_dim: int = 128,
                 mlp_scale: int = 4,
                 output_dim: int = 128,
                 clip_len: int = 4,
                 future_steps: int = 1,
                 num_mlp_layers: int = 2,
                 num_latent_actions: int = 5,
                 feature_size: tuple = (14, 14),
                 feature_to_use: tuple = (1, 0, 0),
                 fc_type: str = 'ft', # st spatio-temporal, s spatial, t temporal
                 num_classes: int = -1,
                 dropout: float = 0.0,
                 component: str = 'verb',
                 norm_type: str = 'bn',
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_dim  = input_dim
        self.hidden_dim = int(input_dim * mlp_scale)
        self.output_dim = output_dim
        self.num_latent_actions = num_latent_actions
        self.feature_to_use = feature_to_use
        self.norm_type = norm_type
        if num_classes == -1:
            self.num_classes = self._num_classes[component]
        else:
            self.num_classes = num_classes
        self.clip_len = clip_len
        self.future_steps = future_steps
        self.num_mlp_layers = num_mlp_layers
        self.feature_size = feature_size
        self.fc_type = fc_type
        self.dropout = dropout
        self.thw = feature_size[0]*feature_size[1]*clip_len
        self.component = component
        self.gen_cls_emb = self.build_mlp()

    def build_mlp(self):
        if self.feature_to_use[0] == 1: # only cur clip feats
            input_dim = self.input_dim * self.clip_len
        elif self.feature_to_use[1] == 1: # cur + futr clip feats
            input_dim = self.input_dim * (self.clip_len * 2)
        elif self.feature_to_use[2] == 1: # cur + futr + la clip feats
            input_dim = self.input_dim * (self.clip_len * 2 + self.num_latent_actions)

        output_dim = self.output_dim * self.num_classes

        layers = []
        for i in range(self.num_mlp_layers-1):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            if self.norm_type == 'bn':
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            elif self.norm_type == 'ln':
                layers.append(nn.LayerNorm(self.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = self.hidden_dim

        if self.num_mlp_layers == 1:
            self.hidden_dim = input_dim

        layers.append(nn.Linear(self.hidden_dim, output_dim))
        if self.norm_type == 'bn':
            layers.append(nn.BatchNorm1d(output_dim))
        elif self.norm_type == 'ln':
            layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        b, t, d = inputs.size() 
        inputs = inputs.reshape([b, t*d])
        
        cls_embs = self.gen_cls_emb(inputs)
        cls_embs = cls_embs.view(b, self.num_classes, -1) # b x c x d

        return cls_embs
    
    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels = torch.cat([sample.cl_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.cl_ivt for sample in data_samples])
        elif self.component == 'tool_tissue':
            labels = torch.cat([sample.cl_it for sample in data_samples])
        elif self.component == 'tool':
            labels = torch.cat([sample.cl_i for sample in data_samples])
        return labels
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList


@MODELS.register_module()
class StateChangeDecoderV2(BaseModule):
    _num_classes = {
        'verb': 10,
        'triplet': 100,
        'instrument-target': 59,
        'instrument': 6,
        'instrument-verb': 26,
    }

    def __init__(self,
                 input_dim: int = 1024,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_classes: int = -1,
                 component: str = 'verb',
                 num_token_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 init_cfg: Optional[ConfigType] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        if num_classes == -1:
            self.num_classes = self._num_classes[component]
        else:
            self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.token_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_token_layers,
        )

        self.class_queries = nn.Parameter(
            torch.randn(1, self.num_classes, hidden_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.class_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        inputs: [B, N, input_dim]
        returns: [B, C, output_dim]
        """
        b, n, _ = inputs.shape

        tokens = self.token_proj(inputs)         # [B, N, H]
        tokens = self.token_encoder(tokens)      # [B, N, H]

        class_queries = self.class_queries.expand(b, -1, -1)   # [B, C, H]
        class_embs, _ = self.cross_attn(
            query=class_queries,
            key=tokens,
            value=tokens,
            need_weights=False
        )                                        # [B, C, H]

        cls_embs = self.class_ffn(class_embs)    # [B, C, output_dim]
        return cls_embs

    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels = torch.cat([sample.cl_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.cl_ivt for sample in data_samples])
        elif self.component == 'instrument-target':
            labels = torch.cat([sample.cl_it for sample in data_samples])
        elif self.component == 'instrument':
            labels = torch.cat([sample.cl_i for sample in data_samples])
        elif self.component == 'instrument-verb':
            labels = torch.cat([sample.cl_iv for sample in data_samples])
        else:
            raise ValueError(f'Unsupported component: {self.component}')
        return labels

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList


@MODELS.register_module()
class PredictiveStateChangeDecoder(BaseModule):
    _num_classes = {
        'verb': 10,
        'triplet': 100,
        'instrument-target': 59,
        'instrument': 6,
        'instrument-verb': 26,
    }

    def __init__(self,
                 input_dim: int = 256,          # centroid dim d
                 hidden_dim: int = 256,
                 output_dim: int = 128,         # class embedding dim
                 pred_hidden_dim: int = 512,
                 num_classes: int = -1,
                 component: str = 'verb',
                 num_token_layers: int = 2,
                 num_attn_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_abs_innovation: bool = True,
                 use_residual_prediction: bool = True,
                 future_loss_weight: float = 1.0,
                 innovation_sparsity_weight: float = 0.05,
                 vicreg_var_weight: float = 25.0,
                 vicreg_cov_weight: float = 1.0,
                 eps: float = 1e-4,
                 init_cfg: Optional[ConfigType] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        if num_classes == -1:
            self.num_classes = self._num_classes[component]
        else:
            self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.component = component
        self.future_loss_weight = future_loss_weight
        self.innovation_sparsity_weight = innovation_sparsity_weight
        self.vicreg_var_weight = vicreg_var_weight
        self.vicreg_cov_weight = vicreg_cov_weight
        self.use_abs_innovation = use_abs_innovation
        self.use_residual_prediction = use_residual_prediction
        self.eps = eps

        change_dim = 4 * input_dim if use_abs_innovation else 3 * input_dim

        # 1) Future predictor from current centroids
        self.future_predictor = nn.Sequential(
            nn.Linear(input_dim, pred_hidden_dim),
            nn.LayerNorm(pred_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden_dim, pred_hidden_dim),
            nn.LayerNorm(pred_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden_dim, input_dim),
        )

        # 2) Change token encoder
        self.token_proj = nn.Sequential(
            nn.Linear(change_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attn_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_token_layers,
        )

        # Per-token gate to suppress noise and highlight informative changes
        self.token_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # 3) Class-query decoder: each class pulls its own evidence
        self.class_queries = nn.Parameter(
            torch.randn(1, self.num_classes, hidden_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.class_refine = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def _variance_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, D]
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(1.0 - std))

    def _covariance_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, D]
        x = x - x.mean(dim=0, keepdim=True)
        n = x.size(0)
        if n <= 1:
            return x.new_tensor(0.0)

        cov = (x.T @ x) / (n - 1)  # [D, D]
        off_diag = cov.flatten()[:-1].view(cov.size(0) - 1, cov.size(1) + 1)[:, 1:].flatten()
        return (off_diag ** 2).mean()

    def _anti_collapse_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] or [B, C, D]
        z = x.reshape(-1, x.size(-1))
        z = F.layer_norm(z, (z.size(-1),))
        var_loss = self._variance_loss(z)
        cov_loss = self._covariance_loss(z)
        return self.vicreg_var_weight * var_loss + self.vicreg_cov_weight * cov_loss

    def _build_change_tokens(
        self,
        cur_ctd: torch.Tensor,
        futr_ctd: torch.Tensor,
        pred_futr: torch.Tensor
    ) -> torch.Tensor:
        innovation = futr_ctd - pred_futr

        feats = [cur_ctd, futr_ctd, pred_futr]
        if self.use_residual_prediction:
            feats[2] = innovation
            feats.insert(2, pred_futr)

        if self.use_abs_innovation:
            feats.append(torch.abs(innovation))

        # default: [cur, futr, innovation, abs(innovation)]
        # if use_residual_prediction=True: [cur, futr, pred_futr, innovation, abs(innovation)]
        # keep 4 blocks by folding prediction into innovation path if desired
        if self.use_residual_prediction:
            change_tokens = torch.cat(
                [cur_ctd, futr_ctd, innovation, torch.abs(innovation)],
                dim=-1
            )
        else:
            change_tokens = torch.cat(feats, dim=-1)

        return change_tokens, innovation

    def forward(
        self,
        inputs: torch.Tensor,
        futr_centroids: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        **kwargs
    ):
        """
        Training:
            inputs = cur_centroids  : [B, N, d]
            futr_centroids          : [B, N, d]

        Inference fallback:
            inputs = cur_centroids
            futr_centroids = None
            then predicted future is used to form self-conditioned change tokens.
        """
        cur_ctd = inputs

        pred_delta = self.future_predictor(cur_ctd)
        pred_futr = cur_ctd + pred_delta if self.use_residual_prediction else pred_delta

        if futr_centroids is None:
            futr_ctd = pred_futr
        else:
            futr_ctd = futr_centroids

        change_tokens, innovation = self._build_change_tokens(cur_ctd, futr_ctd, pred_futr)

        token_feats = self.token_proj(change_tokens)
        token_feats = self.token_encoder(token_feats)

        gate = self.token_gate(token_feats)
        token_feats = token_feats * gate + token_feats

        b = token_feats.size(0)
        class_queries = self.class_queries.expand(b, -1, -1)
        class_feats, _ = self.cross_attn(
            query=class_queries,
            key=token_feats,
            value=token_feats,
            need_weights=False,
        )

        cls_embs = self.class_refine(class_feats)  # [B, C, output_dim]

        if not return_aux:
            return cls_embs

        aux_losses = {}
        if futr_centroids is not None:
            aux_losses['loss_future_pred'] = self.future_loss_weight * F.smooth_l1_loss(pred_futr, futr_centroids)
            aux_losses['loss_innovation_sparse'] = self.innovation_sparsity_weight * innovation.abs().mean()

        aux_losses['loss_token_anticollapse'] = self._anti_collapse_loss(token_feats)
        aux_losses['loss_class_anticollapse'] = self._anti_collapse_loss(cls_embs)

        aux = {
            'pred_futr': pred_futr,
            'innovation': innovation,
            'token_feats': token_feats,
            'aux_losses': aux_losses,
        }
        return cls_embs, aux