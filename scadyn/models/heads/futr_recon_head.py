
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
class FutrReconHead(BaseModule):
    """Future Reconstruction Head to predict future features."""

    def __init__(self,
                 in_channels: int = 384,
                 out_channels: int = 384,
                 clip_len: int = 4,
                 component: str = 'verb',
                 fc_option: str = 'none',
                 loss_feature: dict = None,
                 loss_cls: dict = None,
                 init_cfg: Optional[ConfigType] = None,
                ) -> None:
        super().__init__(init_cfg=init_cfg)

        if loss_feature is not None:
            self.loss_feature = MODELS.build(loss_feature)

    def forward(self, 
                inputs: torch.Tensor, 
                **kwargs) -> dict:
        return inputs
            
    def loss(self,
            future_pred_latent: torch.Tensor, 
            future_latent: torch.Tensor) -> dict:
        losses = dict()
        losses['loss_feature'] = self.loss_feature(future_pred_latent, future_latent)
        return losses
