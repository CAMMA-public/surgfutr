
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class TimeConditionedAnticipationHead(BaseModule):
    """Time Conditioned Anticipation Head for following tasks:
    - Phase Transition
    - Tool Transition
    - IAE Anticipation
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 512,
                 output_dim: int = 128,
                 feat_agg: str = 'pool',
                 clip_len: int = 4,
                 num_centroids: int = 25,
                 feat_type: str = 'centroids',
                 num_sc_classes: int = 10,
                 num_classes: int = 7,
                 normal_clip_ratio: float = 0.2,
                 enable_loss_weight: bool = False,
                 normalize_windows: bool = False,
                 feature_resolution: Tuple[int, int] = (14, 14),
                 max_normal_clip_ratio: float = 0.2,
                 task: str = 'iae-event-antr',
                 component: str = 'verb',
                 loss_task: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.task = task
        self.input_dim = input_dim
        self.component = component
        self.clip_len = clip_len
        self.feat_agg = feat_agg
        self.num_classes = num_classes
        self.num_centroids = num_centroids
        self.num_sc_classes = num_sc_classes
        self.feat_type = feat_type
        self.feature_resolution = feature_resolution
        self.normalize_windows = normalize_windows
        self.enable_loss_weight = enable_loss_weight
        self.max_normal_clip_ratio = max_normal_clip_ratio
        self.loss_task = MODELS.build(loss_task)

        if self.enable_loss_weight:
            if num_classes == 2: # only presence
                # wt = [0.6005, 2.9871] # normal, bleeding
                if self.task == 'cholectrack-bleeding':
                    wt = [1.0612, 0.9455] # using 0.03 max normal clip ratio and normal and bleeding clips are balanced
                    # wt = [1.0041, 0.9960] # using 2, 4, 6 timestamps for background
                else:
                    wt = [1, 1] # normal, bleeding
            elif num_classes == 3: # normal, bleeding1, bleeding2
                if self.task == 'iae-event-antr':
                    if self.max_normal_clip_ratio == 0.2:
                        wt = [0.3918, 4.8292, 4.1568]
                    elif self.max_normal_clip_ratio == 0.1:
                        wt = [0.4503, 2.7745, 2.3882]
                    elif self.max_normal_clip_ratio == 0.05:
                        wt = [0.6667, 1.4412, 1.2405] # 1764 samples
                    elif self.max_normal_clip_ratio == 0.02:
                        wt = [0.9190, 1.1307, 0.9733] # 1344 samples
                    else:
                        wt = [1] * num_classes
                else:
                    wt = [0.7367, 1.2706, 1.1688] # [1] * num_classes
            elif num_classes == 4: # normal, bleeding1, bleeding2, bleeding3
                # wt = [0.3003, 3.7010, 3.1857, 11.7054]
                wt = [0.5015, 1.2353, 1.0633, 3.9070] # NOTE: for 0.04 max normal clip ratio
            elif num_classes == 5: # no cystic interaction, triplet class 68, 69, 78, 79 
                # TODO: change to include other triplets
                wt = [0.26666667, 4.4, 3.09189189, 3.36470588, 2.48695652]
            elif num_classes == 9: # no cystic interaction, triplet class 68, 69, 78, 79 
                wt = [1.0] * 9
            elif num_classes == 7:
                wt = [0.4743, 1.3551, 1.1568, 0.9121, 1.2819, 1.2819, 1.5810]
            elif num_classes == 8:
                # wt = [0.1875, 1.2450, 3.5571, 3.0366, 2.3942, 3.3649, 3.3649, 4.1500]
                # with balancing bg clips
                wt = [1.0472, 0.4712, 1.3464, 1.1494, 0.9062, 1.2736, 1.2736, 1.5708]
            elif num_classes == 6:
                wt = [0.4743, 1.3551, 1.1568, 0.9121, 1.2819, 1.2819, 1.5810]
        else:
            wt = [1] * num_classes

        self.weight = torch.tensor(wt, dtype=torch.float32)

        # embed time 
        self.fc_time_embed = nn.Linear(input_dim+1, input_dim)

        self.setup_classifier()

    def setup_classifier(self):
        if self.feat_type == 'centroids':
            input_dim = self.input_dim * self.num_centroids
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
            input_dim = self.input_dim * self.num_centroids * 2
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
            # inputs = rearrange(inputs, 'b (k h) d -> b k h d', h=self.num_centroids)
            inputs = inputs.mean(dim=2)
        elif self.feat_type == 'centroids_pool':
            # NOTE: also both k and h can be pooled
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2])
        elif self.feat_type == 'imaginator_mean_temporal':
            inputs = inputs.mean(dim=1)
        elif self.feat_type == 'imaginator_flatten':
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
            inputs = inputs.mean(dim=[1, 2, 3]).unsqueeze(1)
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

        # create embedding for time and add to inputs such that input is time-conditioned
        anticipation_windows = torch.cat([sample.anticipation_window for sample in data_samples]) #.to(inputs.device)

        if self.normalize_windows:
            minw, maxw = 1, 5 # data is generated from 1 to 7 seconds
            anticipation_windows = (anticipation_windows - minw) / (maxw - minw)
        
        # repeat the anticipation window to match with the input shape 
        # b x n x d , window shape b x n x 1
        if inputs.ndim == 3:
            anticipation_windows = repeat(anticipation_windows, 'b -> b n 1', n=inputs.shape[1])
        else:
            anticipation_windows = anticipation_windows.unsqueeze(1)

        # add the time conditioned embedding to the input 
        inputs = torch.cat([inputs, anticipation_windows], dim=-1)

        # embed the time conditioned input 
        inputs = self.fc_time_embed(inputs)

        if self.feat_agg == 'pool':
            inputs = inputs.mean(dim=1)
        elif self.feat_agg == 'flatten':
            inputs = inputs.flatten(1, 2)

        preds = self(inputs)
        losses = dict()
        labels = torch.cat([sample.gt_label for sample in data_samples]).long()
        loss_task = self.loss_task(preds.squeeze(-1), labels, weight=self.weight.to(labels.device))
        task_name = data_samples[0].task
        losses[f'loss_{task_name}'] = loss_task

        return losses

    def predict(self, 
                inputs: torch.Tensor, 
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        # labels = torch.cat([sample.gt_label for sample in data_samples]).long()

        if self.feat_type == 'centroids':
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=2)
        elif self.feat_type == 'centroids_pool':
            # NOTE: also both k and h can be pooled
            inputs = rearrange(inputs, 'b (k h) d -> b k h d', k=self.num_centroids)
            inputs = inputs.mean(dim=[1, 2])
        elif self.feat_type == 'imaginator_mean_temporal':
            inputs = inputs.mean(dim=1)
        elif self.feat_type == 'imaginator_flatten': # or self.feat_type == 'imaginator_flatten1':
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
            inputs = inputs.mean(dim=[1, 2, 3]).unsqueeze(1)
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

        anticipation_windows = torch.cat([sample.anticipation_window for sample in data_samples]) #.to(inputs.device)

        if self.normalize_windows:
            minw, maxw = 1, 5
            anticipation_windows = (anticipation_windows - minw) / (maxw - minw)

        if inputs.ndim == 3:
            anticipation_windows = repeat(anticipation_windows, 'b -> b n 1', n=inputs.shape[1])
        else:
            anticipation_windows = anticipation_windows.unsqueeze(1)
        
        inputs = torch.cat([inputs, anticipation_windows], dim=-1)
        inputs = self.fc_time_embed(inputs)

        if self.feat_agg == 'pool':
            inputs = inputs.mean(dim=1)
        elif self.feat_agg == 'flatten':
            inputs = inputs.flatten(1, 2)
        
        preds = self(inputs)
        labels = torch.cat([sample.gt_label for sample in data_samples]).long()

        for data_sample, pred, label in zip(data_samples, preds, labels):
            data_sample.set_pred_score(pred)
            data_sample.set_pred_label(label.unsqueeze(0))

        return data_samples