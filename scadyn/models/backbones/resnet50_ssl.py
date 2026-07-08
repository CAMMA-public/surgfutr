
import torch

from mmaction.registry import MODELS
from mmdet.models.backbones.resnet import ResNet

@MODELS.register_module()
class ResNet50_SSL(ResNet):
    def __init__(self,
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                 **kwargs):
        super().__init__(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            style=style,
            **kwargs)
