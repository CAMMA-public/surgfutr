
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from einops import rearrange

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class RSDPhaseTransitionHead(BaseModule):
    """RSD and Phase Transition Head"""
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 512,
                 output_dim: int = 128,
                 clip_len: int = 4,
                 num_centroids: int = 25,
                 feat_type: str = 'centroids',
                 feat_agg: str = 'pool',
                 num_sc_classes: int = 10,
                 feature_resolution: Tuple[int, int] = (14, 14),
                 num_classes: int = 7,
                 component: str = 'verb',
                 loss_task: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_dim = input_dim
        self.component = component
        self.clip_len = clip_len
        self.feat_agg = feat_agg
        self.num_classes = num_classes
        self.num_centroids = num_centroids
        self.num_sc_classes = num_sc_classes
        self.feat_type = feat_type
        self.feature_resolution = feature_resolution

        self.loss_task = MODELS.build(loss_task)

        self.setup_classifier()

    def setup_classifier(self):
        if self.feat_type == 'centroids':
            input_dim = self.num_centroids * self.input_dim
        elif self.feat_type == 'sc_cls_embs':
            input_dim = self.input_dim * self.num_sc_classes
        elif self.feat_type == 'backbone':
            input_dim = self.input_dim * self.clip_len
        elif self.feat_type == 'imaginator_mean_temporal':
            input_dim = self.input_dim * self.clip_len
        elif self.feat_type == 'imaginator_flatten':
            input_dim = self.input_dim * self.clip_len * 2
        elif self.feat_type == 'imaginator_flatten1':
            input_dim = self.input_dim * self.clip_len
        elif self.feat_type == 'centroids_mean_temporal':
            input_dim = self.input_dim * self.num_centroids
        elif self.feat_type == 'centroids_mean_heads':
            input_dim = self.input_dim * self.num_centroids * 2
        elif self.feat_type == 'centroids_default':
            input_dim = self.input_dim * self.num_centroids
        elif self.feat_type == 'imaginator_default':
            input_dim = self.input_dim * self.clip_len
        elif self.feat_type == 'backbone-tpool':
            input_dim = self.input_dim * self.feature_resolution[0] * self.feature_resolution[1]
        elif self.feat_type == 'backbone-centroids-flatten':
            input_dim = (self.num_centroids + self.clip_len) * self.input_dim
        elif self.feat_type == 'centroids_1':
            input_dim = self.input_dim
        elif self.feat_type == 'centroids_2':
            input_dim = self.input_dim * self.num_centroids
        elif self.feat_type == 'centroids_3':
            pass
        elif self.feat_type == 'centroids_v2_flatten':
            input_dim = self.input_dim * self.num_centroids * 4 # 4 denotes the changes 
        elif self.feat_type == 'centroids_v2_flatten':
            input_dim = self.input_dim
        else:
            input_dim = self.input_dim

        self.fc_eval = nn.Linear(input_dim, self.num_classes)

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        return self.fc_eval(inputs)

    def loss(self, 
            inputs: torch.Tensor, 
            data_samples: SampleList,
            **kwargs) -> dict:
        
        if self.feat_type == 'centroids':
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=2)
        elif self.feat_type == 'centroids_pool':
            # NOTE: also both k and h can be pooled
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2])
        elif self.feat_type == 'imaginator_mean_temporal':
            inputs = inputs.mean(dim=1)
        elif self.feat_type == 'imaginator_flatten' or self.feat_type == 'imaginator_flatten1':
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'centroids_mean_temporal':
            inputs = rearrange(inputs, 'b t (k h) d -> b t k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 3]) # over cur and futr axis (2) and heads
        elif self.feat_type == 'centroids_mean_heads':
            inputs = rearrange(inputs, 'b t (k h) d -> b t k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[3]) # over cur and futr axis (2) and heads
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'backbone-tpool':
            inputs = inputs.mean(dim=[1])
            inputs = inputs.flatten(1, 2) # 14 x 14 for ViT based backbone
        elif self.feat_type == 'backbone-stpool':
            inputs = inputs.mean(dim=[1, 2, 3])
        elif self.feat_type == 'backbone-centroids-pool':
            inputs = inputs.mean(dim=[1])
        elif self.feat_type == 'backbone-centroids-flatten':
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'centroids_1':
            inputs = rearrange(inputs, 'b (k c) d -> b k c d', c=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2]) # b x d
        elif self.feat_type == 'centroids_2':
            inputs = rearrange(inputs, 'b (k c) d -> b k c d', c=self.num_centroids)
            inputs = inputs.mean(dim=[1]) # b x k x d
        elif self.feat_type == 'centroids_3':
            pass
        
        if self.feat_agg == 'pool':
            inputs = inputs.mean(dim=1)
        elif self.feat_agg == 'flatten':
            inputs = inputs.flatten(1, 2)
        else:
            inputs = inputs
        
        preds = self(inputs)
        losses = dict()
        # Keep labels 1-D even for batch size 1 so downstream iteration stays valid.
        labels = torch.cat([sample.gt_label for sample in data_samples]).squeeze(-1)
        loss_task = self.loss_task(preds.squeeze(-1), labels)
        task_name = data_samples[0].task
        losses[f'loss_task_{task_name}'] = loss_task
        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        if self.feat_type == 'centroids':
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=2)
        elif self.feat_type == 'centroids_pool':
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2])
        elif self.feat_type == 'imaginator_mean_temporal':
            inputs = inputs.mean(dim=1)
        elif self.feat_type == 'imaginator_flatten' or self.feat_type == 'imaginator_flatten1':
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'centroids_mean_temporal':
            inputs = rearrange(inputs, 'b t (k h) d -> b t k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 3]) # over cur and futr axis (2) and heads
        elif self.feat_type == 'centroids_mean_heads':
            inputs = rearrange(inputs, 'b t (k h) d -> b t k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[3]) # over cur and futr axis (2) and heads
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'backbone-tpool':
            inputs = inputs.mean(dim=[1])
            inputs = inputs.flatten(1, 2) # 14 x 14 for ViT based backbone
        elif self.feat_type == 'backbone-stpool':
            inputs = inputs.mean(dim=[1, 2, 3])
        elif self.feat_type == 'backbone-centroids-pool':
            inputs = inputs.mean(dim=[1])
        elif self.feat_type == 'backbone-centroids-flatten':
            inputs = inputs.flatten(1, 2)
        elif self.feat_type == 'centroids_1':
            inputs = rearrange(inputs, 'b (k c) d -> b k c d', c=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2]) # b x d
        elif self.feat_type == 'centroids_2':
            inputs = rearrange(inputs, 'b (k c) d -> b k c d', c=self.num_centroids)
            inputs = inputs.mean(dim=[1]) # b x k x d
        elif self.feat_type == 'centroids_3':
            pass

        if self.feat_agg == 'pool':
            inputs = inputs.mean(dim=1)
        elif self.feat_agg == 'flatten':
            inputs = inputs.flatten(1, 2)
        else:
            inputs = inputs
        
        preds = self(inputs)
        labels = torch.cat([sample.gt_label for sample in data_samples]).squeeze(-1)

        for data_sample, pred, label in zip(data_samples, preds, labels):
            data_sample.set_pred_score(pred)
            data_sample.set_pred_label(label.unsqueeze(0))

        return data_samples