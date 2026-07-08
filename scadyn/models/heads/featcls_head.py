
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import multilabel_average_precision
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList

@MODELS.register_module()
class FeatClsHead(BaseModule):
    """Feature Similarity Head to compare the predicted features."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 clip_len: int = 4,
                 component: str = 'verb',
                 fc_option: str = 'none',
                 loss_feature: dict = None,
                 loss_cls: dict = None,
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.component = component
        
        if self.component == 'verb':
            self.num_classes = 10
            self.loss_cls = MODELS.build(dict(
                type='BCELossWithLogits', loss_weight=1.0))
        elif self.component == 'phase':
            self.num_classes = 7
            self.loss_cls = MODELS.build(dict(
            type='CrossEntropyLoss', loss_weight=1.0))
        else:
            # only feature reconstruction loss
            self.num_classes = 1
            loss_cls = None
        
        self.clip_len  = clip_len
        self.fc_option = fc_option

        if loss_feature is not None:
            self.loss_feature = MODELS.build(loss_feature)
        else:
            self.loss_feature = None

        if loss_cls is not None: # phase
            self.loss_cls = MODELS.build(loss_cls)
        else:
            self.loss_cls = None

        self.init_layers()

    def init_layers(self):
        if self.component in ['verb', 'phase']:
            self.fc_cls = nn.Linear(self.input_dim, self.num_classes)
        else:
            self.fc_cls = nn.Identity()

    def forward(self, 
                inputs: torch.Tensor, 
                **kwargs) -> dict:
        logits = self.fc_cls(inputs)
        return logits
            
    def get_mAP(self, pred, gt):
        return multilabel_average_precision(pred, gt, self.num_classes, average='macro')

    def loss(self,
            latent: torch.Tensor,
            future_pred_latent: torch.Tensor, 
            future_latent: torch.Tensor,
            future_mask: torch.Tensor,
            future_steps: int,
            data_samples: SampleList) -> dict:
        losses = dict()
        
        # feature matching loss
        # future_latent = future_latent.detach()

        if self.loss_feature is not None:
            if future_mask is not None:
                future_mask = future_mask.unsqueeze(-1).expand_as(future_latent)
                future_pred_latent = future_pred_latent * future_mask
                future_latent = future_latent * future_mask
            losses['loss_feature'] = self.loss_feature(future_pred_latent, future_latent)

        if self.loss_cls is not None:
            current_logits = self(latent.mean(dim=1))

            if self.component == 'verb':
                labels = torch.cat([sample.cl_v for sample in data_samples])
            elif self.component == 'phase':
                labels = torch.cat([sample.phase for sample in data_samples]).long()
            losses['loss_cls'] = self.loss_cls(current_logits, labels)

        # # compute auxillary predictions
        # if self.component == 'verb':
        #     fc_labels = torch.cat([sample.futr_v for sample in data_samples]).long()
        #     with torch.no_grad():
        #         gt_future_logits = self(future_latent.mean(dim=1)).sigmoid()
        #         losses['aux_gtfutr_mAP'] = self.get_mAP(gt_future_logits, fc_labels)
        #         pred_future_logits = self(future_pred_latent.mean(dim=1)).sigmoid()
        #         losses['aux_predfutr_mAP'] = self.get_mAP(pred_future_logits, fc_labels)
        #         losses['aux_feat_sim'] = F.cosine_similarity(future_pred_latent, future_latent, dim=1).mean()
        return losses

    def predict(self, 
            latent: torch.Tensor,
            future_pred_latent: torch.Tensor, 
            future_latent: torch.Tensor,
            future_mask: torch.Tensor,
            future_steps: int,
            data_samples: SampleList) -> SampleList:
        
        if self.loss_cls is not None:
            current_logits = self(latent.mean(dim=1))
            if self.component == 'phase':
                current_score  = current_logits.softmax(dim=-1)
                labels = torch.cat([sample.phase for sample in data_samples]).long()
            elif self.component == 'verb':
                current_score  = current_logits.sigmoid()
                labels = torch.cat([sp.cl_v for sp in data_samples])

            # for current clip classification and labels
            for data_sample, score, label in zip(data_samples, current_score, labels):
                data_sample.set_pred_score(score)
                data_sample.set_pred_label(label)

            # with torch.no_grad():
            #     gt_future_logits = self(future_latent.mean(dim=1)).sigmoid().detach()
            #     pred_future_logits = self(future_pred_latent.mean(dim=1)).sigmoid().detach()
            #     fc_labels = torch.cat([sample.futr_v for sample in data_samples])

            # # for future clip classification and labels
            # for data_sample, flogit, pflogit, label in zip(data_samples, gt_future_logits, 
            #                                             pred_future_logits, fc_labels):
            #     data_sample.set_value_x(flogit, 'gt_future_logits')
            #     data_sample.set_value_x(pflogit, 'pred_future_logits')
            #     data_sample.set_value_x(label, 'fc_labels')

        if self.loss_feature is not None:
            feats = InstanceData() # sim between pred futr & gt futr feats
            # future_mask = future_mask.unsqueeze(-1).bool() # apply mask 
            # b, t, d = future_pred_latent.shape
            # future_pred_latent = torch.masked_select(future_pred_latent, future_mask).view(b, -1, d)
            # future_latent = torch.masked_select(future_latent, future_mask).view(b, -1, d)

            feats.pred_future  = future_pred_latent.detach()
            feats.gt_future    = future_latent.detach()
        else:
            feats = None

        return data_samples, feats


