"""
Reimplementation of the SOTA action anticipation models such as AVT, RULSTM, SlowFastAnticipation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from mmaction.models.recognizers import Recognizer2D
from mmaction.utils import (ConfigType,OptConfigType,SampleList)
from mmaction.registry import MODELS
from einops import rearrange
from ..utils.ssl_model_layers import (
    create_r50_ssl_backbone, create_avt_model, create_rulstm_model, RULSTMTemporal)

@MODELS.register_module()
class BaselineActionAnticipator(Recognizer2D):
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 state_change_decoder: OptConfigType = None,
                 clustering: OptConfigType = None,
                 graph: OptConfigType = None,
                 imaginator: OptConfigType = None,
                 centroid_transition: OptConfigType = None,
                 aux_head: OptConfigType = None,
                 use_imaginator=False,
                 model_name='AVT',
                 future_steps: int = 4,
                 slowfast_ratio: int = 2,
                 freeze_backbone=False,
                 backbone_ckpt: str = '',
                 agg_feats='mean',
                 use_futr=True,
                 layer_idx=23,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.epoch = 0
        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.use_futr = use_futr
        self.agg_feats = agg_feats
        self.backbone_ckpt = backbone_ckpt
        self.future_steps = future_steps
        self.slowfast_ratio = max(1, int(slowfast_ratio))
        self.model_name = model_name
        self.avt_feature_dim = 2048
        self.with_avt = False
        self.with_rulstm = False
        self.with_rulstm_pretrain = False
        self.rulstm_feature_dim = 1024
        self.rulstm_dropout_p = 0.8
        self.rulstm_pretrain_fc = None
        self.with_slowfast_rulstm = False
        self.with_slowfast_rulstm_pretrain = False


        if backbone_ckpt != '' and backbone_type == 'ResNet50_SSL':
            self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        elif backbone_ckpt == 'AVT':
            # get modules for AVT such as encoder, decoder, cls head
            num_classes = 0
            if hasattr(self, 'cls_head') and self.cls_head is not None:
                num_classes = int(
                    getattr(self.cls_head, 'num_classes',
                            getattr(self.cls_head, 'num_states', 0)))
            backbone, decoder = create_avt_model(num_classes=num_classes)
            self.backbone = backbone
            self.decoder = decoder
            self.with_avt = True
            in_dim = int(getattr(self.backbone, 'embed_dim', 768))
            self.avt_feature_proj = nn.Identity() if in_dim == self.avt_feature_dim else nn.Linear(in_dim, self.avt_feature_dim)
        elif backbone_ckpt in ['RULSTM', 'RULSTM_PRETRAIN']:
            # NOTE: the pretraining part is adapted from original work to process per frame 
            self.backbone = create_rulstm_model(
                model_name='inception_v3.tv_in1k',
                pretrained=False,
                output_dim=1024,
                ckpt_name='weights/inception_v3_tv_in1k.bin')
            if backbone_ckpt == 'RULSTM_PRETRAIN':
                self.with_rulstm_pretrain = True
            else:
                self.with_rulstm = True
                self.rulstm_temporal = RULSTMTemporal(
                    input_dim=self.rulstm_feature_dim,
                    hidden_dim=self.rulstm_feature_dim,
                    anticipation_steps=max(1, self.future_steps),
                    dropout_p=self.rulstm_dropout_p,
                )
                self.rulstm_cls_dropout = nn.Dropout(self.rulstm_dropout_p)
        elif backbone_ckpt in ['SLOWFAST_RULSTM', 'SLOWFAST_RULSTM_PRETRAIN']:
            # SlowFast RU-LSTM: two temporal branches over the same RGB features
            # sampled at different rates, then fused with learned attention weights.
            self.backbone = create_rulstm_model(
                model_name='inception_v3.tv_in1k',
                pretrained=False,
                output_dim=1024,
                ckpt_name='weights/inception_v3_tv_in1k.bin')
            if backbone_ckpt == 'SLOWFAST_RULSTM_PRETRAIN':
                self.with_slowfast_rulstm_pretrain = True
            else:
                self.with_slowfast_rulstm = True
                self.rulstm_temporal_fast = RULSTMTemporal(
                    input_dim=self.rulstm_feature_dim,
                    hidden_dim=self.rulstm_feature_dim,
                    anticipation_steps=max(1, self.future_steps),
                    dropout_p=self.rulstm_dropout_p,
                )
                self.rulstm_temporal_slow = RULSTMTemporal(
                    input_dim=self.rulstm_feature_dim,
                    hidden_dim=self.rulstm_feature_dim,
                    anticipation_steps=max(1, self.future_steps),
                    dropout_p=self.rulstm_dropout_p,
                )
                self.rulstm_cls_dropout = nn.Dropout(self.rulstm_dropout_p)
                self.slowfast_fusion_mlp = nn.Linear(self.rulstm_feature_dim * 2, 2)
            
        if self.freeze_backbone:
            self.backbone.requires_grad = False
            # self.backbone.patch_embed.requires_grad = False # TODO: turn off during training ?
            print("Freezing backbone")

        self.with_imaginator = imaginator is not None
        if self.with_imaginator:
            self.imaginator = MODELS.build(imaginator)

    def preprocess_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == 'MViT':
            patch_tokens, cls_token, last_tokens = inputs[0], inputs[1], inputs[-1]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type in ['VisionTransformer']:
            patch_tokens, last_tokens = inputs
        return patch_tokens, last_tokens
    
    def get_backbone_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == 'ResNet50_SSL':
            stacked_feats_2d = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame) # b x c x h x w
                stacked_feats_2d.append(feats)
            stacked_feats_2d = torch.stack(stacked_feats_2d, dim=2)
            latent = rearrange(stacked_feats_2d, 'b c t h w -> b t h w c')
        elif self.backbone_type == 'AVT':
            stacked_feats_2d = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame) # b x c x h x w NOTE: only using the cls token feats as mentioned in AVT
                stacked_feats_2d.append(feats)
            stacked_feats_2d = torch.stack(stacked_feats_2d, dim=2)
            latent = rearrange(stacked_feats_2d, 'b d l -> b l d')    
        elif self.backbone_type == 'RULSTM':
            # inputs: [B, C, T, H, W] -> latent: [B, T, D]
            frame_feats = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame)  # [B, 1024]
                frame_feats.append(feats)
            latent = torch.stack(frame_feats, dim=1)  # [B, T, 1024]
        else:
            features = self.backbone(inputs)
            latent, last_tokens = self.preprocess_features(features)

        if self.with_neck:
            latent = self.neck(latent)
        return latent

    def aggregate_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if self.agg_feats == 'mean':
                return features.mean(dim=1)
            elif self.agg_feats == 'sum':
                return features.sum(dim=1)
            elif self.agg_feats == 'flatten':
                return features.flatten(1, 2)
        else:
            return features

    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        if inputs.size(1) == 2:
            cur_inputs, futr_inputs = torch.split(inputs, 1, dim=1)
        else:
            cur_inputs = inputs
            futr_inputs = inputs.clone()

        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        futr_inputs = futr_inputs.view((-1, ) + futr_inputs.shape[2:])
         
        losses = dict()
        cur_vec = self.get_backbone_features(cur_inputs)

        # AVT path: z_t -> causal decoder -> z_hat_t
        if self.with_avt:
            # cur_vec: [B, T, D_enc] (e.g. D_enc=768)
            z = self.avt_feature_proj(cur_vec)  # [B, T, 2048]
            z_hat = self.decoder(z, return_logits=False, use_last_only=False)  # [B, T, 2048]

            # L_feat = sum_{t=1}^{T-1} ||z_hat_t - z_{t+1}||^2
            if z_hat.size(1) > 1:
                losses['loss_feat'] = F.mse_loss(z_hat[:, :-1], z[:, 1:].detach())

            # L_next from AVTHead (per-class new/not_new classification).
            next_feat = z_hat[:, -1]  # [B, D]
            head_losses = self.cls_head.loss(next_feat, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses

        if self.with_rulstm_pretrain:
            # Stage-1 pretraining: TSN-style current clip verb recognition.
            # Aggregate snippet features and supervise with current-clip verb labels (`cl_v`).
            pretrain_losses = self.cls_head.loss(cur_vec, data_samples)
            # losses['loss_cls'] = F.binary_cross_entropy_with_logits(logits, labels)
            losses.update(pretrain_losses)
            return losses

        if self.with_slowfast_rulstm_pretrain:
            # Stage-1 pretraining: same recognition target, used before anticipation fine-tuning.
            pretrain_losses = self.cls_head.loss(cur_vec, data_samples)
            losses.update(pretrain_losses)
            return losses

        if self.with_rulstm:
            # RU-LSTM: R-LSTM encoding + U-LSTM unrolling, then use last-step anticipated feature.
            z_hat = self.rulstm_temporal(cur_vec)  # [B, T, D]
            next_feat = self.rulstm_cls_dropout(z_hat[:, -1])  # [B, D]
            head_losses = self.cls_head.loss(next_feat, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses

        if self.with_slowfast_rulstm:
            # Fast branch over all frames.
            fast_seq = self.rulstm_temporal_fast(cur_vec)  # [B, T, D]
            fast_feat = self.rulstm_cls_dropout(fast_seq[:, -1])  # [B, D]

            # Slow branch over sub-sampled frames.
            slow_in = cur_vec[:, ::self.slowfast_ratio, :]
            if slow_in.size(1) == 0:
                slow_in = cur_vec[:, -1:, :]
            slow_seq = self.rulstm_temporal_slow(slow_in)  # [B, T_s, D]
            slow_feat = self.rulstm_cls_dropout(slow_seq[:, -1])  # [B, D]

            # Attention-style fusion weights from slow/fast branch states.
            gate_logits = self.slowfast_fusion_mlp(torch.cat([slow_feat, fast_feat], dim=-1))  # [B, 2]
            gate = torch.softmax(gate_logits, dim=-1)
            fused_feat = gate[:, :1] * slow_feat + gate[:, 1:] * fast_feat  # [B, D]

            head_losses = self.cls_head.loss(fused_feat, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses

        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        if inputs.size(1) == 2:
            cur_inputs, futr_inputs = torch.split(inputs, 1, dim=1)
        else:
            cur_inputs  = inputs
            futr_inputs = inputs.clone()

        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        futr_inputs = futr_inputs.view((-1, ) + futr_inputs.shape[2:])

        cur_vec = self.get_backbone_features(cur_inputs)

        if self.with_avt:
            z = self.avt_feature_proj(cur_vec)
            z_hat = self.decoder(z, return_logits=False, use_last_only=False)
            next_feat = z_hat[:, -1]  # [B, D]
            return self.cls_head.predict(next_feat, data_samples)

        if self.with_rulstm_pretrain:
            return self.cls_head.predict(cur_vec, data_samples)

        if self.with_rulstm:
            z_hat = self.rulstm_temporal(cur_vec)  # [B, T, D]
            next_feat = self.rulstm_cls_dropout(z_hat[:, -1])  # [B, D]
            return self.cls_head.predict(next_feat, data_samples)

        if self.with_slowfast_rulstm_pretrain:
            return self.cls_head.predict(cur_vec, data_samples)

        if self.with_slowfast_rulstm:
            fast_seq = self.rulstm_temporal_fast(cur_vec)  # [B, T, D]
            fast_feat = self.rulstm_cls_dropout(fast_seq[:, -1])  # [B, D]

            slow_in = cur_vec[:, ::self.slowfast_ratio, :]
            if slow_in.size(1) == 0:
                slow_in = cur_vec[:, -1:, :]
            slow_seq = self.rulstm_temporal_slow(slow_in)  # [B, T_s, D]
            slow_feat = self.rulstm_cls_dropout(slow_seq[:, -1])  # [B, D]

            gate_logits = self.slowfast_fusion_mlp(torch.cat([slow_feat, fast_feat], dim=-1))  # [B, 2]
            gate = torch.softmax(gate_logits, dim=-1)
            fused_feat = gate[:, :1] * slow_feat + gate[:, 1:] * fast_feat
            return self.cls_head.predict(fused_feat, data_samples)

        return self.cls_head.predict(cur_vec, data_samples)
