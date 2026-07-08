
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)
from einops import rearrange

class AttentionPooledClassifier(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_heads: int, 
                 num_classes: int, 
                 num_states: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, num_classes * num_states, bias=False)

    def forward(self, inputs):
        b, l, d = inputs.size()
        q = inputs
        k = inputs
        v = inputs
        attn_out, _ = self.attn(q, k, v)
        return self.fc(attn_out)

class ResidualMLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm(x)
        x = F.gelu(x)  # GELU instead of ReLU
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual

@MODELS.register_module()
class StateChangeHead(BaseModule):
    """State Change Head to predict state change from the class embeddings."""
    def __init__(self,
                 feature_dim: int=384,
                 input_dim: int=128,
                 hidden_dim: int=512,
                 output_dim: int=384,
                 seq_len: int=25,
                 num_states: int=4,
                 num_heads: int=8,
                 num_classes: int=10,
                 num_centroids: int=2,
                 method: int = 1,
                 pool_cur_futr: bool = False,
                 use_class_weights: bool = True,
                 component: str = 'verb',
                 downsample_feature: bool = True,
                 loss_state: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_states = num_states
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.component = component
        self.seq_len = seq_len
        self.pool_cur_futr = pool_cur_futr
        self.method = method
        self.num_centroids = num_centroids
        self.downsample_feature = downsample_feature
        self.use_class_weights = use_class_weights
        self.loss_type = loss_state['type']
        self.loss_cls = MODELS.build(loss_state)

        # self.downsample_fc = nn.Sequential(
        #     nn.Linear(self.feature_dim, self.input_dim),
        #     nn.LayerNorm(self.input_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.input_dim, self.input_dim))
        if self.downsample_feature:
            self.downsample_fc = nn.Linear(self.feature_dim, self.input_dim)
        else:
            self.downsample_fc = nn.Identity()

        self.build_state_classifier()

    def build_state_classifier(self):
        if self.method == 1: # direct mlp
            self.state_classifier = nn.Linear(self.seq_len * self.input_dim, self.num_classes * self.num_states)
        elif self.method == 2: # mlp with hidden dim
            self.state_classifier = nn.Sequential(
                nn.Linear(self.seq_len * self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.num_classes * self.num_states)
            )
        elif self.method == 3: # attention based use multihead attention
            self.state_classifier = AttentionPooledClassifier(self.input_dim, self.num_heads, self.num_classes, self.num_states)
        elif self.method == 4: # residual mlp
            self.state_classifier = nn.Sequential(
                nn.LayerNorm(self.input_dim * self.seq_len),
                ResidualMLP(self.input_dim * self.seq_len, self.hidden_dim, dropout=0.1),
                nn.Linear(self.input_dim * self.seq_len, self.num_classes * self.num_states)
            )
        elif self.method in [5, 6]:
            self.state_classifier = nn.Sequential(
                nn.LayerNorm(self.input_dim * self.seq_len),
                nn.Linear(self.input_dim * self.seq_len, self.num_classes * self.num_states))
        else:
            raise ValueError(f'Invalid method: {self.method}')

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        inputs = self.downsample_fc(inputs)

        b, l, d = inputs.size()
        if self.pool_cur_futr and l != self.num_centroids and self.num_centroids > 1:
            inputs = rearrange(inputs, 'b (p k) d -> b p k d', p=self.num_centroids)
            inputs = inputs.mean(dim=1)
        elif self.pool_cur_futr and self.num_centroids == 0:
            inputs = inputs.mean(dim=1)

        if self.method in [1, 2, 4, 5, 6]:
            inputs = inputs.flatten(1)

        return self.state_classifier(inputs)
    
    def get_weights(self, device: torch.device):
        if self.use_class_weights:
            if self.component == 'verb':
                wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
            elif self.component == 'triplet':
                wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
        else:
            wt = [1.0, 1.0, 1.0, 1.0]
        return torch.tensor(wt).to(device)
    
    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels = torch.cat([sample.sc_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'tool_tissue':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'tool':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:

        state_logits = self(inputs)
        state_logits = rearrange(state_logits, 'b (c s) -> b c s', c=self.num_classes, s=self.num_states)

        losses = dict()
        labels = self.get_labels(data_samples)
        class_weight = self.get_weights(inputs.device)

        loss_sc = self.loss_cls(state_logits.permute(0, 2, 1), labels, weight=class_weight)
        losses[f'loss_sc_{self.component}'] = loss_sc

        losses['state_logits'] = state_logits

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        state_logits = self(inputs)
        state_logits = rearrange(state_logits, 'b (c s) -> b c s', c=self.num_classes, s=self.num_states)

        state_logits = state_logits.softmax(dim=-1)
        labels = self.get_labels(data_samples)

        for data_sample, logit, label in zip(data_samples, state_logits, labels):
            data_sample.set_pred_score(logit)
            data_sample.set_pred_label(label)

        return data_samples