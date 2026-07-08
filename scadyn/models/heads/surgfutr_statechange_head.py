
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmengine.structures import InstanceData
from mmaction.utils import (ConfigType, SampleList)
from einops import rearrange
from torchmetrics.functional.classification import multilabel_average_precision

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
class SurgFutrStateChangeHead(BaseModule):
    """SurgFutr State Change Head to predict state change."""
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
                 component: str = 'verb',
                 dataset: str = 'cholect50',
                 center: str = 's',
                 downsample_feature: bool = True,
                 loss_state: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_feature: dict = dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=1.0),
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
        self.dataset = dataset
        self.center = center
        self.loss_type = loss_state['type']

        self.loss_cls = MODELS.build(loss_state)
        self.loss_feature = MODELS.build(loss_feature)

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
            self.state_classifier = nn.Linear(self.input_dim, self.num_states)

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

    def get_mAP(self, pred, gt):
        return multilabel_average_precision(pred, gt, self.num_classes, average='macro')
    
    def get_weights(self, device: torch.device):
        # if self.component == 'verb':
        #     wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
        # elif self.component == 'triplet':
        #     wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
        # return torch.tensor(wt).to(device)
    
        if self.dataset == 'cholect50':
            if self.component == 'verb':
                wt = [1.9524793388429753, 9.218555417185554, 8.544632550981147, 0.30653020097173145]
            elif self.component == 'triplet':
                wt = [18.72154779969651, 77.67576075550892, 73.02696481420585, 0.25510086865823206]
            elif self.component == 'instrument':
                wt = [1.1214965070280278, 10.14041095890411, 9.409957627118644, 0.3444180215576292]
        elif self.dataset == 'prosta':
            if self.component == 'verb':
                wt = [1.578496832734121, 3.1925207756232687, 2.940051020408163, 0.36857885268838697]
            elif self.component == 'triplet':
                wt = [11.636131593874078, 16.577373737373737, 17.077627471383973, 0.2634920125230794]
        elif self.dataset == 'grasp':
            wt = [8.188049853372434, 14.442025862068965, 14.59298780487805, 0.26737236063009717]
        elif self.dataset == 'mbp':
            if self.center == 's':
                wt = [16.386742756804214, 38.29924760601915, 38.53647625602202, 0.2572734123009346]
            elif self.center == 'ac':
                wt = [17.06734693877551, 34.94712936046512, 35.17721287490856, 0.257442240823177]
            elif self.center == 'b':
                wt = [18.115870153291254, 31.148062015503875, 31.36690085870414, 0.2576778786032642]
            elif self.center == 's1':
                wt = [16.43384929906542, 38.30462899931926, 38.30462899931926, 0.25727198741747287]
            elif self.center == 'ac1':
                wt = [17.13809271054494, 34.906126126126125, 34.93130183916336, 0.25744170223623153]
            elif self.center == 'b1':
                wt = [18.220377019748653, 31.083460949464012, 31.131134969325153, 0.25767731779461983]
        return torch.tensor(wt).to(device)
    
    def get_labels(self, data_samples: SampleList) -> torch.Tensor:
        if self.component == 'verb':
            labels = torch.cat([sample.sc_v for sample in data_samples])
        elif self.component == 'triplet':
            labels = torch.cat([sample.sc_ivt for sample in data_samples])
        elif self.component == 'instrument_tissue':
            labels = torch.cat([sample.sc_it for sample in data_samples])
        elif self.component == 'instrument':
            labels = torch.cat([sample.sc_i for sample in data_samples])
        return labels

    def loss(self, 
            inputs: torch.Tensor, 
            future_pred_latent: torch.Tensor,
            future_latent: torch.Tensor,
            future_mask: torch.Tensor,
            future_steps: int,
            data_samples: SampleList,
            **kwargs) -> dict:

        state_logits = self(inputs)
        if self.method in [1, 2, 4, 5, 6]:
            state_logits = rearrange(state_logits, 'b (c s) -> b c s', c=self.num_classes, s=self.num_states)

        losses = dict()
        labels = self.get_labels(data_samples)
        class_weight = self.get_weights(inputs.device)

        loss_sc = self.loss_cls(state_logits.permute(0, 2, 1), labels, weight=class_weight)
        losses[f'loss_sc_{self.component}'] = loss_sc
        losses['state_logits'] = state_logits

        # future prediction loss
        losses['loss_feature'] = self.loss_feature(future_pred_latent, future_latent.detach())

        # compute auxillary predictions
        if self.component == 'verb' or self.component == 'instrument':
            # fc_labels = torch.cat([sample.futr_v for sample in data_samples]).long()
            with torch.no_grad():
                #gt_future_logits = self(future_latent.mean(dim=1)).sigmoid()
                #losses['aux_gtfutr_mAP'] = self.get_mAP(gt_future_logits, fc_labels)
                #pred_future_logits = self(future_pred_latent.mean(dim=1)).sigmoid()
                #losses['aux_predfutr_mAP'] = self.get_mAP(pred_future_logits, fc_labels)
                losses['aux_feat_sim'] = F.cosine_similarity(future_pred_latent, future_latent, dim=1).mean()

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                future_pred_latent: torch.Tensor,
                future_latent: torch.Tensor,
                future_mask: torch.Tensor,
                future_steps: int,
                data_samples: SampleList,
                **kwargs) -> SampleList:
        state_logits = self(inputs)
        if self.method in [1, 2, 4, 5, 6]:
            state_logits = rearrange(state_logits, 'b (c s) -> b c s', c=self.num_classes, s=self.num_states)

        state_logits = state_logits.softmax(dim=-1)
        labels = self.get_labels(data_samples)

        for data_sample, logit, label in zip(data_samples, state_logits, labels):
            data_sample.set_pred_score(logit)
            data_sample.set_pred_label(label)

        feats = InstanceData()
        feats.pred_future = future_pred_latent.detach()
        feats.gt_future = future_latent.detach()

        if self.component == 'verb' or self.component == 'instrument':
            return data_samples, feats
        return data_samples