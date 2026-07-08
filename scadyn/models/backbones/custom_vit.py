from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix, _load_checkpoint

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import VisionTransformer, Block, Attention

class CustomAttention(Attention):
    def __init__(self,
                 embed_dims: int,   
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 **kwargs) -> None:
        super().__init__(embed_dims=embed_dims,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop_rate=attn_drop_rate,
                         drop_rate=drop_rate,
                         **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, 'q_bias'):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # for use in OT
        self.qkv_tokens = attn # b x num_heads x thw x thw

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # self.qkv_tokens = x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomBlock(Block):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 **kwargs) -> None:
        super().__init__(embed_dims=embed_dims,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop_rate=attn_drop_rate,
                         drop_rate=drop_rate,
                         **kwargs)
        self.attn = CustomAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate)

@MODELS.register_module()
class CustomViT(VisionTransformer):
    def __init__(self,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_frames: int = 8,
                 num_heads: int = 12,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
                 init_values: int = 0.,
                 return_feat_map: bool = False,
                 return_feat_before_norm: bool = False,
                 **kwargs) -> None:

        super().__init__(
            embed_dims=embed_dims,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            init_values=init_values,
            **kwargs
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ModuleList([
            CustomBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(depth)
        ])

        self.return_feat_map = return_feat_map
        self.return_feat_before_norm = return_feat_before_norm

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size,
                                               self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h, w), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            qkv_tokens = blk.attn.qkv_tokens # extra output for OT # NOTE: consider only n-1 or n-2 blocks
            
        if self.return_feat_before_norm:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            return x, qkv_tokens

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            # x = x.permute(0, 4, 1, 2, 3)
            return x, qkv_tokens

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1)) # NOTE: activates only when mean pooling is used!

        return x[:, 0]
    

# @MODELS.register_module()
# class CustomViT_HF(nn.Module):
#     def __init__(self,
#                  embed_dims: int = 768,
#                  depth: int = 12,
#                  num_frames: int = 8,
#                  num_heads: int = 12,
#                  mlp_ratio: int = 4.,
#                  qkv_bias: bool = True,
#                  qk_scale: int = None,
#                  drop_rate: float = 0.,
#                  attn_drop_rate: float = 0.,
#                  drop_path_rate: float = 0.,
#                  norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
#                  init_values: int = 0.,
#                  return_feat_map: bool = False,
#                  return_feat_before_norm: bool = False,
#                  **kwargs) -> None:
#         super().__init__(**kwargs)

#     def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
#         pass


