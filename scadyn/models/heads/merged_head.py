
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, SampleList)

@MODELS.register_module()
class MergedAnticipationHead(BaseModule):
    """Time Conditioned Anticipation Head for following tasks:
    - Phase Transition
    - Tool Transition
    - IAE Anticipation
    """

    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 512,
                 output_dim: int = 128,
                 feature_dim: int = 384,
                 feat_agg: str = 'pool',
                 clip_len: int = 4,
                 num_centroids: int = 25,
                 feat_type: str = 'centroids',
                 num_sc_classes: int = 10,
                 num_classes: int = 7,
                 max_normal_clip_ratio: float = 0.2,
                 enable_loss_weight: bool = False,
                 normalize_windows: bool = False,
                 task: str = 'reg',
                 component: str = 'verb',
                 loss_task: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.component = component
        self.clip_len = clip_len
        self.feat_agg = feat_agg
        self.num_classes = num_classes
        self.num_centroids = num_centroids
        self.num_sc_classes = num_sc_classes
        self.feat_type = feat_type
        self.normalize_windows = normalize_windows
        self.enable_loss_weight = enable_loss_weight
        self.max_normal_clip_ratio = max_normal_clip_ratio
        self.loss_task = MODELS.build(loss_task)

        if self.enable_loss_weight:
            if num_classes == 2: # only presence
                wt = [0.6005, 2.9871] # normal, bleeding
            elif num_classes == 3: # normal, bleeding1, bleeding2
                if self.max_normal_clip_ratio == 0.2:
                    wt = [0.3918, 4.8292, 4.1568]
                elif self.max_normal_clip_ratio == 0.1:
                    wt = [0.4503, 2.7745, 2.3882]
                elif self.max_normal_clip_ratio == 0.05:
                    wt = [0.6667, 1.4412, 1.2405] # 1764 samples
            elif num_classes == 4: # normal, bleeding1, bleeding2, bleeding3
                wt = [0.3003, 3.7010, 3.1857, 11.7054]
            elif num_classes == 5: # no cystic interaction, triplet class 68, 69, 78, 79 
                # TODO: change to include other triplets
                wt = [0.26666667, 4.4, 3.09189189, 3.36470588, 2.48695652]
            elif num_classes == 9: # no cystic interaction, triplet class 68, 69, 78, 79 
                wt = [1.0] * 9
        else:
            wt = [1] * num_classes

        self.weight = torch.tensor(wt, dtype=torch.float32)

        if self.feat_type == 'merge_1':
            input_dim = self.input_dim + self.feature_dim
        elif self.feat_type == 'merge_2':
            input_dim = 4352
        elif self.feat_type == 'merge_3':
            input_dim = 7680
        elif self.feat_type == 'merge_31':
            input_dim = 1536

        # embed time 
        self.fc_time_embed = nn.Linear(input_dim+1, input_dim)

        self.setup_classifier()

    def setup_classifier(self):
        if self.feat_type == 'merge_1':
            input_dim = 2 * 4 * (self.input_dim + self.feature_dim)
        elif self.feat_type == 'merge_2':
            input_dim = 4352
        elif self.feat_type == 'merge_3':
            input_dim = 7680
        elif self.feat_type == 'merge_31':
            input_dim = 1536
        else:
            input_dim = self.input_dim

        self.fc_eval = nn.Linear(input_dim, self.num_classes)

    def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
        return self.fc_eval(inputs)

    def loss(self,
            sc_class_embs: torch.Tensor,
            centroids_before_sc: torch.Tensor,
            cur_vec_pooled: torch.Tensor,
            future_pred_vec: torch.Tensor,
            data_samples: SampleList,
            **kwargs) -> dict:
        
        # breakpoint()

        if self.feat_type == 'merge_1':
            feat_stacked = torch.stack([cur_vec_pooled, future_pred_vec], dim=1)
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=2)
            inputs = torch.cat([feat_stacked, centroids_before_sc], dim=-1)
            inputs = inputs.flatten(1, 2) # b x 2 x 4 x (256 + 384)
        elif self.feat_type == 'merge_2': # curfutr feats +  sc emb
            feat_stacked = torch.stack([cur_vec_pooled, future_pred_vec], dim=1).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([feat_stacked, sc_class_embs], dim=-1) # b x 4352
        elif self.feat_type == 'merge_3':
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=[1, 3]).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([centroids_before_sc, sc_class_embs], dim=-1) # b x 7680
        elif self.feat_type == 'merge_31':
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=[1, 2, 3]).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([centroids_before_sc, sc_class_embs], dim=-1) # b x 1536

        # create embedding for time and add to inputs such that input is time-conditioned
        anticipation_windows = torch.cat([sample.anticipation_window for sample in data_samples]) #.to(inputs.device)

        if self.normalize_windows:
            minw, maxw = 1, 5 # 7 as dataset is only using till 5 seconds
            anticipation_windows = (anticipation_windows - minw) / (maxw - minw)
        
        # repeat the anticipation window to match with the input shape 
        # b x n x d , window shape b x n x 1
        if self.feat_type in ['merge_2', 'merge_3', 'merge_31']:
            anticipation_windows = anticipation_windows.unsqueeze(1)
        else:
            anticipation_windows = repeat(anticipation_windows, 'b -> b n 1', n=inputs.shape[1])

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
        try:
            loss_task = self.loss_task(preds.squeeze(-1), labels, weight=self.weight.to(labels.device))
        except:
            breakpoint()
        task_name = data_samples[0].task
        losses[f'loss_{task_name}'] = loss_task

        # compute f1 score using torchmetrics function
        # f1 = multiclass_f1_score(preds.detach(), labels.detach(), num_classes=self.num_classes, average='macro')
        # losses['f1'] = f1

        return losses

    def predict(self,
                sc_class_embs: torch.Tensor,
                centroids_before_sc: torch.Tensor,
                cur_vec_pooled: torch.Tensor,
                future_pred_vec: torch.Tensor,
                data_samples: SampleList,
                **kwargs) -> SampleList:
        
        if self.feat_type == 'merge_1':
            feat_stacked = torch.stack([cur_vec_pooled, future_pred_vec], dim=1)
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=2)
            inputs = torch.cat([feat_stacked, centroids_before_sc], dim=-1)
            inputs = inputs.flatten(1, 2) # b x 2 x 4 x (256 + 384)
        elif self.feat_type == 'merge_2': # curfutr feats +  sc emb
            feat_stacked = torch.stack([cur_vec_pooled, future_pred_vec], dim=1).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([feat_stacked, sc_class_embs], dim=-1) # b x 4352
        elif self.feat_type == 'merge_3':
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=[1, 3]).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([centroids_before_sc, sc_class_embs], dim=-1) # b x 7680
        elif self.feat_type == 'merge_31':
            centroids_before_sc = rearrange(centroids_before_sc, 'b p (k h) d -> b p k h d', k=self.num_centroids, p=2)
            centroids_before_sc = centroids_before_sc.mean(dim=[1, 2, 3]).flatten(1)
            sc_class_embs = sc_class_embs.flatten(1)
            inputs = torch.cat([centroids_before_sc, sc_class_embs], dim=-1) # b x 1536

        anticipation_windows = torch.cat([sample.anticipation_window for sample in data_samples]) #.to(inputs.device)

        if self.normalize_windows:
            minw, maxw = 1, 5
            anticipation_windows = (anticipation_windows - minw) / (maxw - minw)

        if self.feat_type in ['merge_2', 'merge_3', 'merge_31']:
            anticipation_windows = anticipation_windows.unsqueeze(1)
        else:
            anticipation_windows = repeat(anticipation_windows, 'b -> b n 1', n=inputs.shape[1])

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