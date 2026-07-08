from typing import Optional, Union, Tuple, Dict, List
import torch
import torch.nn.functional as F
from mmengine.model import ModuleList
from torch import Tensor
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
import numpy as np
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmaction.models.backbones.mvit import (MultiScaleBlock, MultiScaleAttention,
                                            attention_pool, add_decomposed_rel_pos,
                                            resize_pos_embed, MViT)

class CustomMultiScaleAttention(MultiScaleAttention):
    def __init__(self,
                 in_dims: int,
                 out_dims: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 norm_cfg: Dict = dict(type='LN'),
                 pool_kernel: Tuple[int] = (3, 3, 3),
                 stride_q: Tuple[int] = (1, 1, 1),
                 stride_kv: Tuple[int] = (1, 1, 1),
                 rel_pos_embed: bool = True,
                 residual_pooling: bool = True,
                 input_size: Optional[Tuple[int]] = None,
                 rel_pos_zero_init: bool = False,
                 with_cls_token: bool = True,                 
                 **kwargs) -> None:
        super().__init__(in_dims=in_dims,
                         out_dims=out_dims,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         pool_kernel=pool_kernel,
                         stride_q=stride_q,
                         stride_kv=stride_kv,
                         rel_pos_embed=rel_pos_embed,
                         residual_pooling=residual_pooling,
                         input_size=input_size,
                         rel_pos_zero_init=rel_pos_zero_init,
                         with_cls_token=with_cls_token,
                         **kwargs)

    def forward(self, x: torch.Tensor, in_size: Tuple[int]) -> tuple:
        """Forward the MultiScaleAttention."""
        B, N, _ = x.shape  # (B, H*W, C)

        # qkv: (B, H*W, 3, num_heads, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1)
        # q, k, v: (B, num_heads, H*W, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            in_size,
            norm=self.norm_q,
            with_cls_token=self.with_cls_token)
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            in_size,
            norm=self.norm_k,
            with_cls_token=self.with_cls_token)
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            in_size,
            norm=self.norm_v,
            with_cls_token=self.with_cls_token)
        
        attn = (q * self.scale) @ k.transpose(-2, -1) # first block out b x 1 x 393 x 96
        if self.rel_pos_embed:
            attn = add_decomposed_rel_pos(attn, q, q_shape, k_shape,
                                          self.rel_pos_h, self.rel_pos_w,
                                          self.rel_pos_t, self.with_cls_token)

        attn = attn.softmax(dim=-1) # B, 1, 12545, 197 first block
        x = attn @ v

        # attention valued features
        self.values = x.clone()
        self.attn_matrix = attn.clone()
        self.key_features = k.clone()

        if self.residual_pooling:
            if self.with_cls_token:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        # (B, num_heads, H'*W', C'//num_heads) -> (B, H'*W', C')
        x = x.transpose(1, 2).reshape(B, -1, self.out_dims)
        x = self.proj(x)

        return x, q_shape

class CustomMultiScaleBlock(MultiScaleBlock):
    def __init__(self,
            in_dims: int,
            out_dims: int,
            num_heads: int,
            qkv_bias: bool = True,
            norm_cfg: Dict = dict(type='LN'),
            qkv_pool_kernel: Tuple = (3, 3, 3),
            stride_q: Tuple = (1, 1, 1),
            stride_kv: Tuple = (1, 1, 1),
            rel_pos_embed: bool = True,
            residual_pooling: bool = True,
            with_cls_token: bool = True,
            dim_mul_in_attention: bool = True,
            input_size: Optional[Tuple[int]] = None,
            rel_pos_zero_init: bool = False,
            **kwargs) -> None:
        super().__init__(in_dims=in_dims,
                         out_dims=out_dims,
                         num_heads=num_heads,
                         qkv_bias=qkv_bias,
                         norm_cfg=norm_cfg,
                         qkv_pool_kernel=qkv_pool_kernel,
                         stride_q=stride_q,
                         stride_kv=stride_kv,
                         rel_pos_embed=rel_pos_embed,
                         residual_pooling=residual_pooling,
                         with_cls_token=with_cls_token,
                         dim_mul_in_attention=dim_mul_in_attention,
                         input_size=input_size,
                         rel_pos_zero_init=rel_pos_zero_init,
                         **kwargs)

        attn_dims = out_dims if dim_mul_in_attention else in_dims
        self.attn = CustomMultiScaleAttention(
            in_dims,
            attn_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pooling=residual_pooling,
            input_size=input_size,
            rel_pos_zero_init=rel_pos_zero_init,
            with_cls_token=with_cls_token)

@MODELS.register_module()
class CustomMViTv2(MViT):
    def __init__(self,
        drop_path_rate: float = 0.,
        pool_kernel: tuple = (3, 3, 3),
        adaptive_kv_stride: tuple = (1, 8, 8),
        dim_mul: int = 2,
        head_mul: int = 2,
        rel_pos_embed: bool = True,
        residual_pooling: bool = True,
        dim_mul_in_attention: bool = True,
        with_cls_token: bool = True,
        rel_pos_zero_init: bool = False,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        norm_cfg: Dict = dict(type='LN', eps=1e-6),
        **kwargs) -> None:

        super().__init__(
            drop_path_rate=drop_path_rate,
            pool_kernel=pool_kernel,
            adaptive_kv_stride=adaptive_kv_stride,
            dim_mul=dim_mul,
            head_mul=head_mul,
            rel_pos_embed=rel_pos_embed,
            residual_pooling=residual_pooling,
            dim_mul_in_attention=dim_mul_in_attention,
            with_cls_token=with_cls_token,
            rel_pos_zero_init=rel_pos_zero_init,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            **kwargs
        )

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.blocks = ModuleList()
        out_dims_list = [self.embed_dims]
        num_heads = self.num_heads
        stride_kv = adaptive_kv_stride
        input_size = self.patch_resolution
        for i in range(self.num_layers):
            if i in self.downscale_indices or i in self.dim_mul_indices:
                num_heads *= head_mul

            if i in self.downscale_indices:
                stride_q = [1, 2, 2]
                stride_kv = [max(s // 2, 1) for s in stride_kv]
            else:
                stride_q = [1, 1, 1]
            
            # Set output embed_dims
            if dim_mul_in_attention and i in self.dim_mul_indices:
                # multiply embed_dims in dim_mul_indices layers.
                out_dims = out_dims_list[-1] * dim_mul
            elif not dim_mul_in_attention and i + 1 in self.dim_mul_indices:
                # multiply embed_dims before dim_mul_indices layers.
                out_dims = out_dims_list[-1] * dim_mul
            else:
                out_dims = out_dims_list[-1]

            attention_block = CustomMultiScaleBlock(
                in_dims=out_dims_list[-1],
                out_dims=out_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                qkv_pool_kernel=pool_kernel,
                stride_q=stride_q,
                stride_kv=stride_kv,
                rel_pos_embed=rel_pos_embed,
                residual_pooling=residual_pooling,
                with_cls_token=with_cls_token,
                dim_mul_in_attention=dim_mul_in_attention,
                input_size=input_size,
                rel_pos_zero_init=rel_pos_zero_init)
            self.blocks.append(attention_block)

            input_size = attention_block.init_out_size
            out_dims_list.append(out_dims)

            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    norm_layer = build_norm_layer(norm_cfg, out_dims)[1]
                    self.add_module(f'norm{stage_index}', norm_layer)

    def forward(self, x: torch.Tensor) ->\
            Tuple[Union[torch.Tensor, List[torch.Tensor]]]:
        """Forward the MViT."""
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, block in enumerate(self.blocks):
            x, patch_resolution = block(x, patch_resolution)
            attn_values = block.attn.values
            attn_matrix = block.attn.attn_matrix 
            key_features = block.attn.key_features

            if i in self.stage_indices:
                stage_index = self.stage_indices[i]
                if stage_index in self.out_scales:
                    B, _, C = x.shape
                    x = getattr(self, f'norm{stage_index}')(x)
                    tokens = x.transpose(1, 2)
                    if self.with_cls_token:
                        patch_token = tokens[:, :, 1:].reshape(
                            B, C, *patch_resolution)
                        cls_token = tokens[:, :, 0]
                    else:
                        patch_token = tokens.reshape(B, C, *patch_resolution)
                        cls_token = None
                    if self.output_cls_token:
                        out = [patch_token, cls_token]
                    else:
                        out = patch_token
                    outs.append(out)

        # if cls token enabled, then the first element of outs is cls_token
        if self.output_cls_token:
            attn_matrix = attn_matrix[:, :, 1:, 1:] # attn from patch tokens to patch tokens

        return outs[0][0], outs[0][-1], attn_matrix
