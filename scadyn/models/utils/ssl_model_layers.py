"""
Load up pretrained SSL models - R50 (Moco/DINO) and ViT based (EndoViT)
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models import resnet50, resnet101, resnet152, densenet169
from mmengine.runner import load_checkpoint

from timm.models.vision_transformer import VisionTransformer
from functools import partial
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoModel
from typing import Optional, Dict


def _raise_missing_backbone_helper(helper_name: str) -> None:
    raise NotImplementedError(
        f'`{helper_name}` is referenced by recognizer modules but is not '
        'implemented in `scadyn.models.utils.ssl_model_layers`. Restore the '
        'original helper or avoid configs that request this backbone.')


def create_internvideo2_backbone(backbone, freeze_backbone=True):
    _raise_missing_backbone_helper('create_internvideo2_backbone')


def create_dinov3_backbone(backbone_ckpt=None, freeze_backbone=True):
    _raise_missing_backbone_helper('create_dinov3_backbone')


def create_surgvista_backbone(backbone_ckpt=None, freeze_backbone=True):
    _raise_missing_backbone_helper('create_surgvista_backbone')


def create_vjepa2_backbone(backbone_ckpt=None, freeze_backbone=True):
    _raise_missing_backbone_helper('create_vjepa2_backbone')


class VJEPA2AttentionExtractor(nn.Module):
    def __init__(self, backbone, layer_idx=23):
        super().__init__()
        _raise_missing_backbone_helper('VJEPA2AttentionExtractor')

def set_to_identity(model, layer_list):
    for layer in layer_list:
        module = model
        layer_name = layer.split('.')

        if len(layer_name) == 1:
            name = layer_name[0]
        else:
            name = layer_name[-1]
            for l in layer_name[:-1]:
                if l.isdigit():
                    module = module[int(l)]
                else:
                    module = getattr(module, l)
        setattr(module, name, torch.nn.Identity())
    return model

def freeze_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        return FrozenBatchNorm2d(module.num_features)
    for name, child in module.named_children():
        module.add_module(name, freeze_bn(child))
    return module

def create_r50_ssl_backbone(backbone, backbone_ckpt=None, freeze_backbone=True):
    assert backbone_ckpt is not None, "backbone_ckpt must be provided"
    _ = load_checkpoint(backbone, backbone_ckpt, strict=False)
    backbone._is_init = True

    # set layers in backbone to identity
    set_to_identity(backbone, ['fc'])

    backbone = torch.nn.Sequential(*list(backbone.children())[:-1]) # NOTE: ignored as mmdet ResNet only has backbone layers

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        freeze_bn(backbone)

    return backbone

def create_endovit_backbone(backbone_ckpt=None, freeze_backbone=True):
    # model_path = snapshot_download(repo_id="egeozsoy/EndoViT", revision="main")
    model_weights_path = 'weights/endovit_pytorch_model.bin'
    if backbone_ckpt == '-1':
        # model_weights_path = Path(model_path) / "pytorch_model.bin"
        model_weights = torch.load(model_weights_path)['model']
    # else:
    #     model_weights = torch.load(backbone_ckpt)['model'] #TODO: verify this

    backbone = VisionTransformer(patch_size=16, 
                                embed_dim=768, 
                                depth=12, 
                                num_heads=12, 
                                mlp_ratio=4, 
                                qkv_bias=True, 
                                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    backbone.load_state_dict(model_weights, strict=False)
    backbone.head = torch.nn.Identity()

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    print("Successfully Loaded EndoViT model ..")
    return backbone

from ..backbones.custom_metaformer import caformer_s18
def create_surgenet_backbone(backbone_ckpt=None, freeze_backbone=False):
    model_weights_path = 'weights/SurgeNetXL_checkpoint_epoch0050_teacher.pth'
    backbone = caformer_s18(pretrained_weights=model_weights_path)
    backbone.head = torch.nn.Identity()

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    print("Successfully Loaded SurgeNetXL model ..")
    return backbone

class RULSTMFrameEncoder(nn.Module):
    """RU-LSTM RGB representation extractor.

    Uses an Inception-style timm backbone and returns per-frame features.
    """

    def __init__(
        self,
        model_name: str = "inception_v3.tv_in1k",
        pretrained: bool = False,
        output_dim: int = 1024,
        ckpt_name: str = "weights/inception_v3_tv_in1k.bin",
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg')
        if ckpt_name is not None:
            if not os.path.isfile(ckpt_name):
                raise FileNotFoundError(
                    f"RULSTM local checkpoint not found: {ckpt_name}. "
                    "Place timm Inception weights there or update ckpt_name.")
            state_dict = torch.load(ckpt_name, map_location='cpu')
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            if isinstance(state_dict, dict):
                cleaned = {}
                for key, value in state_dict.items():
                    nk = key
                    if nk.startswith('module.'):
                        nk = nk[len('module.'):]
                    if nk.startswith('encoder.'):
                        nk = nk[len('encoder.'):]
                    cleaned[nk] = value
                state_dict = cleaned
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Successfully loaded local timm Inception weights:", ckpt_name)
        self.input_size = int(self.encoder.default_cfg.get('input_size', (3, 299, 299))[1])
        in_dim = int(getattr(self.encoder, 'num_features', 2048))
        self.output_dim = int(output_dim)
        self.proj = nn.Identity() if in_dim == self.output_dim else nn.Linear(in_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D]
        if x.ndim != 4:
            raise ValueError(f"RULSTMFrameEncoder expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False)
        feats = self.encoder(x)
        return self.proj(feats)


class RSDNetFrameEncoder(nn.Module):
    """RSDNet RGB representation extractor using ResNet-152."""

    def __init__(
        self,
        pretrained: bool = False,
        output_dim: int = 2048,
        ckpt_name: str | None = "weights/resnet152_imagenet1k.pth",
    ) -> None:
        super().__init__()
        try:
            # Torchvision>=0.13 style
            from torchvision.models import ResNet152_Weights
            weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            self.encoder = resnet152(weights=weights)
        except Exception:
            # Backward compatible fallback
            self.encoder = resnet152(pretrained=pretrained)

        # Keep global average pooled backbone feature (pre-classifier), 2048-D.
        self.encoder.fc = nn.Identity()

        if ckpt_name is not None and os.path.isfile(ckpt_name):
            state_dict = torch.load(ckpt_name, map_location='cpu')
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            if isinstance(state_dict, dict):
                cleaned = {}
                for key, value in state_dict.items():
                    nk = key
                    if nk.startswith('module.'):
                        nk = nk[len('module.'):]
                    if nk.startswith('encoder.'):
                        nk = nk[len('encoder.'):]
                    cleaned[nk] = value
                state_dict = cleaned
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Successfully loaded local ResNet-152 weights:", ckpt_name)
        elif ckpt_name is not None:
            print(f"Warning: local ResNet-152 checkpoint not found at {ckpt_name}. Using current initialization.")

        self.input_size = 224
        in_dim = 2048
        self.output_dim = int(output_dim)
        self.proj = nn.Identity() if in_dim == self.output_dim else nn.Linear(in_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RSDNetFrameEncoder expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False)
        feats = self.encoder(x)
        return self.proj(feats)


class TransLocalFrameEncoder(nn.Module):
    """TransLocal-style frame encoder using ResNet-101."""

    def __init__(
        self,
        pretrained: bool = False,
        output_dim: int = 2048,
        ckpt_name: str | None = "weights/resnet101_imagenet1k.pth",
    ) -> None:
        super().__init__()
        try:
            # Torchvision>=0.13 style
            from torchvision.models import ResNet101_Weights
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            self.encoder = resnet101(weights=weights)
        except Exception:
            # Backward compatible fallback
            self.encoder = resnet101(pretrained=pretrained)

        self.encoder.fc = nn.Identity()

        if ckpt_name is not None and os.path.isfile(ckpt_name):
            state_dict = torch.load(ckpt_name, map_location='cpu')
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            if isinstance(state_dict, dict):
                cleaned = {}
                for key, value in state_dict.items():
                    nk = key
                    if nk.startswith('module.'):
                        nk = nk[len('module.'):]
                    if nk.startswith('encoder.'):
                        nk = nk[len('encoder.'):]
                    cleaned[nk] = value
                state_dict = cleaned
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Successfully loaded local ResNet-101 weights:", ckpt_name)
        elif ckpt_name is not None:
            print(f"Warning: local ResNet-101 checkpoint not found at {ckpt_name}. Using current initialization.")

        self.input_size = 224
        in_dim = 2048
        self.output_dim = int(output_dim)
        self.proj = nn.Identity() if in_dim == self.output_dim else nn.Linear(in_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"TransLocalFrameEncoder expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False)
        feats = self.encoder(x)
        return self.proj(feats)


class BDNetFrameEncoder(nn.Module):
    """BD-Net frame encoder: DenseNet169 with RGB+elapsed-time channel."""

    def __init__(
        self,
        pretrained: bool = False,
        output_dim: int = 1664,
        ckpt_name: str | None = "weights/densenet169_imagenet1k.pth",
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import DenseNet169_Weights
            weights = DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = densenet169(weights=weights)
        except Exception:
            self.encoder = densenet169(pretrained=pretrained)

        # Convert first conv to 4 channels (RGB + elapsed-time map).
        conv0 = self.encoder.features.conv0
        old_w = conv0.weight.data.clone()
        new_conv0 = nn.Conv2d(
            4,
            conv0.out_channels,
            kernel_size=conv0.kernel_size,
            stride=conv0.stride,
            padding=conv0.padding,
            bias=False,
        )
        new_conv0.weight.data[:, :3] = old_w
        new_conv0.weight.data[:, 3] = old_w.mean(dim=1)
        self.encoder.features.conv0 = new_conv0

        if ckpt_name is not None and os.path.isfile(ckpt_name):
            state_dict = torch.load(ckpt_name, map_location='cpu')
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
            if isinstance(state_dict, dict):
                cleaned = {}
                for key, value in state_dict.items():
                    nk = key
                    if nk.startswith('module.'):
                        nk = nk[len('module.'):]
                    if nk.startswith('encoder.'):
                        nk = nk[len('encoder.'):]
                    cleaned[nk] = value
                state_dict = cleaned
            self.encoder.load_state_dict(state_dict, strict=False)
            print("Successfully loaded local DenseNet169 weights:", ckpt_name)
        elif ckpt_name is not None:
            print(f"Warning: local DenseNet169 checkpoint not found at {ckpt_name}. Using current initialization.")

        self.encoder.classifier = nn.Identity()
        self.input_size = 224
        in_dim = 1664
        self.output_dim = int(output_dim)
        self.proj = nn.Identity() if in_dim == self.output_dim else nn.Linear(in_dim, self.output_dim)

    def forward(self, x: torch.Tensor, elapsed_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"BDNetFrameEncoder expects [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False)

        b, _, h, w = x.shape
        if elapsed_norm is None:
            et = torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)
        else:
            et = elapsed_norm.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
            et = et.expand(-1, 1, h, w)

        x4 = torch.cat([x, et], dim=1)
        feats = self.encoder(x4)
        return self.proj(feats)

import timm
class AVTDecoder(nn.Module):
    """AVT-style causal temporal decoder (GPT-like).

    Input: frame features z of shape [B, T, D_in]
    Output: predicted future features z_hat of shape [B, T, D_model]
    Optionally decodes z_hat to class logits via a linear head.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_tokens: int = 16,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.max_tokens = max_tokens

        self.proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, model_dim))
        self.drop = nn.Dropout(dropout)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # GPT-style stack: masked self-attention blocks + final LN.
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _causal_mask(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Upper-triangular mask blocks attending to future positions.
        mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        use_last_only: bool = False,
    ) -> torch.Tensor:
        # x: [B, D] or [B, T, D]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"AVTDecoder expects [B, D] or [B, T, D], got {tuple(x.shape)}")

        b, t, _ = x.shape
        if t > self.max_tokens:
            raise ValueError(f"Token length {t} exceeds max_tokens={self.max_tokens}")

        x = self.proj(x)
        x = x + self.pos_embed[:, :t]
        x = self.drop(x)
        x = self.decoder(x, mask=self._causal_mask(t, x.device, x.dtype))
        x = self.final_norm(x)

        if return_logits:
            logits = self.classifier(x)
            return logits[:, -1] if use_last_only else logits
        return x[:, -1] if use_last_only else x


class AVTFrameEncoder(nn.Module):
    """Frame-wise AVT encoder with shared weights across frames.

    Matches the AVT description:
    - patch-level tokens + prepended learnable cls token
    - learned spatial position embeddings
    - pre-norm transformer encoder
    - temporal handling is intentionally delegated to the head/decoder
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        patch_flat_dim: int = 256,
        init_from_timm: str | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_flat_dim = patch_flat_dim
        self.use_timm = init_from_timm is not None

        if self.use_timm:
            self.vit = timm.create_model(
                init_from_timm,
                pretrained=pretrained,
                num_classes=0,
            )
            ckpt_name = 'weights/vit_base_patch16_224_orig_in21k_ft_in1k.bin'
            # load the checkpoint
            state_dict = torch.load(ckpt_name, map_location='cpu')
            self.vit.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained ViT from timm:", init_from_timm)
            self.num_patches = self.vit.patch_embed.num_patches
            return

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Linear(patch_flat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, N, patch_flat_dim]
        b, c, h, w = x.shape
        p = self.patch_size
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"Expected {(self.img_size, self.img_size)}, got {(h, w)}")

        patches = F.unfold(x, kernel_size=p, stride=p)  # [B, C*p*p, N]
        # AVT text describes 256D flattened patches; keep channel-averaged 16x16 vectors.
        patches = patches.view(b, c, p * p, -1).mean(dim=1)  # [B, 256, N]
        return patches.transpose(1, 2).contiguous()  # [B, N, 256]

    def forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D]
        if self.use_timm:
            feats = self.vit.forward_features(x)
            return feats if feats.ndim == 2 else feats[:, 0]

        x = self._patchify(x)
        x = self.patch_proj(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D]
        # [B, T, C, H, W] -> [B, T, D]
        if x.ndim == 4:
            return self.forward_frame(x)
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            z = self.forward_frame(x)
            return z.view(b, t, -1)
        raise ValueError(f"AVTFrameEncoder expects 4D/5D input, got {tuple(x.shape)}")


class AVTCausalClipEncoder(nn.Module):
    """Causal feature extractor for spatiotemporal backbones.

    Ensures z_t uses only clip frames [t-L+1, ..., t] (no future leakage).
    """

    def __init__(self, clip_backbone: nn.Module, clip_len: int = 8):
        super().__init__()
        self.clip_backbone = clip_backbone
        self.clip_len = clip_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C, H, W] -> [B, T, D]
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        out = []
        for ti in range(t):
            start = max(0, ti - self.clip_len + 1)
            clip = x[:, start : ti + 1]
            if clip.size(1) < self.clip_len:
                pad = clip[:, :1].expand(-1, self.clip_len - clip.size(1), -1, -1, -1)
                clip = torch.cat([pad, clip], dim=1)
            clip = clip.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, L, H, W]
            z = self.clip_backbone(clip)
            if z.ndim > 2:
                z = z.flatten(1)
            out.append(z)
        return torch.stack(out, dim=1)
    
def create_avt_model(
    variant: str = "avt_b",
    num_classes: int = 0,
    clip_len: int = 10,
):
    """Create AVT backbone + decoder.

    AVT-b: 12 heads, 12 layers, 768D; IN21k->IN1k ViT-B init.
    AVT-h: 4 heads, 6 layers, 2048D; from-scratch style setup.
    """
    variant = variant.lower()
    if variant not in {"avt_b", "avt_h"}:
        raise ValueError(f"Unsupported AVT variant: {variant}")

    if variant == "avt_b":
        backbone = AVTFrameEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            patch_flat_dim=256,
            init_from_timm="vit_base_patch16_224.orig_in21k_ft_in1k",
            pretrained=False,
        )
    else:
        backbone = AVTFrameEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=2048,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            patch_flat_dim=256,
            init_from_timm=None,
            pretrained=False,
        )

    decoder = AVTDecoder(
        input_dim=2048,
        model_dim=2048,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
        max_tokens=clip_len,
        num_classes=num_classes,
    )
    return backbone, decoder


def create_rulstm_model(
    model_name: str = "inception_v3.tv_in1k",
    pretrained: bool = False,
    output_dim: int = 1024,
    ckpt_name: str = "weights/inception_v3_tv_in1k.bin",
):
    """Create RU-LSTM RGB backbone (frame-level feature extractor)."""
    backbone = RULSTMFrameEncoder(
        model_name=model_name,
        pretrained=pretrained,
        output_dim=output_dim,
        ckpt_name=ckpt_name,
    )
    return backbone


def create_rsdnet_model(
    pretrained: bool = False,
    output_dim: int = 2048,
    ckpt_name: str | None = "weights/resnet152_imagenet1k.pth",
):
    """Create RSDNet RGB backbone (ResNet-152 frame encoder)."""
    backbone = RSDNetFrameEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        ckpt_name=ckpt_name,
    )
    return backbone


def create_translocal_model(
    pretrained: bool = False,
    output_dim: int = 2048,
    ckpt_name: str | None = "weights/resnet101_imagenet1k.pth",
):
    """Create TransLocal RGB backbone (ResNet-101 frame encoder)."""
    backbone = TransLocalFrameEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        ckpt_name=ckpt_name,
    )
    return backbone


def create_bdnet_model(
    pretrained: bool = False,
    output_dim: int = 1664,
    ckpt_name: str | None = "weights/densenet169_imagenet1k.pth",
):
    """Create BD-Net RGB+time backbone (DenseNet169 frame encoder)."""
    backbone = BDNetFrameEncoder(
        pretrained=pretrained,
        output_dim=output_dim,
        ckpt_name=ckpt_name,
    )
    return backbone

class RULSTMTemporal(nn.Module):
    """RU-LSTM temporal module with R-LSTM encoder and U-LSTM unrolling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        anticipation_steps: int = 1,
        dropout_p: float = 0.8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else input_dim)
        self.anticipation_steps = int(max(1, anticipation_steps))
        self.r_input_drop = nn.Dropout(dropout_p)
        self.u_input_drop = nn.Dropout(dropout_p)
        self.r_lstm = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.u_lstm = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.out_proj = nn.Identity() if self.hidden_dim == self.input_dim else nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D]
        if feats.ndim != 3:
            raise ValueError(f"RULSTMTemporal expects [B, T, D], got {tuple(feats.shape)}")

        b, t, d = feats.shape
        if d != self.input_dim:
            raise ValueError(f"Expected D={self.input_dim}, got D={d}")

        h_r = feats.new_zeros((b, self.hidden_dim))
        c_r = feats.new_zeros((b, self.hidden_dim))
        pred_feats = []

        for ti in range(t):
            f_t = feats[:, ti]
            f_t_r = self.r_input_drop(f_t)
            h_r, c_r = self.r_lstm(f_t_r, (h_r, c_r))

            # nt mirrors RU-LSTM unrolling schedule: larger unroll for earlier timesteps.
            nt = self.anticipation_steps + (t - 1 - ti)
            h_u, c_u = h_r, c_r
            for _ in range(nt):
                f_t_u = self.u_input_drop(f_t)
                h_u, c_u = self.u_lstm(f_t_u, (h_u, c_u))

            pred_feats.append(self.out_proj(h_u))

        return torch.stack(pred_feats, dim=1)  # [B, T, D]
