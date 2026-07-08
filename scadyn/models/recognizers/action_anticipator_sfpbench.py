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
    create_r50_ssl_backbone, create_avt_model, create_rulstm_model, create_rsdnet_model, create_translocal_model, create_bdnet_model, RULSTMTemporal)

@MODELS.register_module()
class BaselineActionAnticipatorSFPBench(Recognizer2D):
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
                 head_feature_src: str = 'backbone',
                 agg_feats='mean',
                 use_futr=True,
                 bdnet_dncl_weight: float = 0.0,
                 bdnet_num_blstms: int = 5,
                 bdnet_dropouts: Optional[List[float]] = None,
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
        self.head_feature_src = head_feature_src
        self.with_rulstm = False
        self.with_rulstm_pretrain = False
        self.with_slowfast_rulstm = False
        self.with_slowfast_rulstm_pretrain = False
        self.rulstm_feature_dim = 1024
        self.rulstm_dropout_p = 0.8
        self.rulstm_pretrain_fc = None
        self.with_rsdnet = False
        self.rsdnet_feature_dim = 2048
        self.rsdnet_hidden_dim = 1024
        self.rsdnet_dropout_p = 0.5
        self.with_bdnet = False
        self.bdnet_feature_dim = 1664
        self.bdnet_hidden_dim = 128
        self.bdnet_dropout_p = 0.2
        self.bdnet_elapsed_scale = 300.0
        self.bdnet_dncl_weight = float(bdnet_dncl_weight)
        self.bdnet_num_blstms = int(max(1, bdnet_num_blstms))
        self.bdnet_dropouts_cfg = bdnet_dropouts
        self.with_timelstm = False
        self.timelstm_feature_dim = 2048
        self.with_translocal = False
        self.translocal_feature_dim = 2048


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
            # SlowFast RU-LSTM: fast + slow temporal branches and learned fusion gate.
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
        elif backbone_ckpt == 'RSDNET':
            # RSDNet: ResNet-152 frame encoder + LSTM temporal model.
            self.backbone = create_rsdnet_model(
                pretrained=False,
                output_dim=self.rsdnet_feature_dim,
                ckpt_name='weights/resnet152_imagenet1k.pth')
            self.with_rsdnet = True
            self.rsdnet_temporal = nn.LSTM(
                input_size=self.rsdnet_feature_dim,
                hidden_size=self.rsdnet_hidden_dim,
                num_layers=1,
                batch_first=True)
            self.rsdnet_dropout = nn.Dropout(self.rsdnet_dropout_p)
        elif backbone_ckpt == 'BDNET':
            # BD-Net: DenseNet169 with RGB+elapsed-time channel + LSTM temporal model.
            self.backbone = create_bdnet_model(
                pretrained=False,
                output_dim=self.bdnet_feature_dim,
                ckpt_name='weights/densenet169_imagenet1k.pth')
            self.with_bdnet = True
            # Ensemble of stochastic LSTM regressors following BD-Net style.
            if self.bdnet_dropouts_cfg is not None and len(self.bdnet_dropouts_cfg) > 0:
                self.bdnet_branch_dropouts = [float(p) for p in self.bdnet_dropouts_cfg]
            else:
                # Default follows the released BD-Net code.
                default_ps = [0.0, 0.2, 0.4, 0.4, 0.6]
                if self.bdnet_num_blstms == len(default_ps):
                    self.bdnet_branch_dropouts = default_ps
                else:
                    self.bdnet_branch_dropouts = [self.bdnet_dropout_p] * self.bdnet_num_blstms
            self.bdnet_input_dropouts = nn.ModuleList([
                nn.Dropout(p=p) for p in self.bdnet_branch_dropouts
            ])
            self.bdnet_temporals = nn.ModuleList([
                nn.LSTM(
                    input_size=self.bdnet_feature_dim,
                    hidden_size=self.bdnet_hidden_dim,
                    num_layers=1,
                    batch_first=True)
                for _ in self.bdnet_branch_dropouts
            ])
        elif backbone_ckpt == 'TIMELSTM':
            # TimeLSTM phase-pretraining backbone (ResNet-152).
            self.backbone = create_rsdnet_model(
                pretrained=False,
                output_dim=self.timelstm_feature_dim,
                ckpt_name='weights/resnet152_imagenet1k.pth')
            self.with_timelstm = True
        elif backbone_ckpt == 'TRANSLOCAL':
            # TransLocal phase-pretraining backbone (ResNet-101).
            self.backbone = create_translocal_model(
                pretrained=False,
                output_dim=self.translocal_feature_dim,
                ckpt_name='weights/resnet101_imagenet1k.pth')
            self.with_translocal = True
            
        if self.freeze_backbone:
            self.backbone.requires_grad = False
            # self.backbone.patch_embed.requires_grad = False # TODO: turn off during training ?
            print("Freezing backbone")

        self.with_imaginator = imaginator is not None
        if self.with_imaginator:
            self.imaginator = MODELS.build(imaginator)

    def _bdnet_ensemble_forward(self, cur_vec: torch.Tensor):
        branch_feats = []
        for input_drop, temporal in zip(self.bdnet_input_dropouts, self.bdnet_temporals):
            z_hat, _ = temporal(input_drop(cur_vec))  # [B, T, D]
            branch_feats.append(z_hat)
        mean_feat = torch.stack(branch_feats, dim=0).mean(dim=0)
        return mean_feat, branch_feats

    def _bdnet_diversity_loss(self, branch_feats):
        if len(branch_feats) <= 1:
            return None
        last_feats = [bf[:, -1] for bf in branch_feats]  # [B, D] per branch
        norms = [F.normalize(f, dim=-1) for f in last_feats]
        sims = []
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                sims.append((norms[i] * norms[j]).sum(dim=-1).pow(2).mean())
        if not sims:
            return None
        return torch.stack(sims, dim=0).mean()

    def preprocess_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == 'MViT':
            patch_tokens, cls_token, last_tokens = inputs[0], inputs[1], inputs[-1]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type in ['VisionTransformer', 'InternVideo2', 'CustomSwin', 'CustomUniFormerV2']:
            patch_tokens, last_tokens = inputs
        return patch_tokens, last_tokens
    
    def _extract_clip_meta(self, sample) -> Optional[torch.Tensor]:
        if hasattr(sample, "clip_meta"):
            return torch.as_tensor(sample.clip_meta, dtype=torch.float32)
        if hasattr(sample, "metainfo") and isinstance(sample.metainfo, dict):
            if "clip_meta" in sample.metainfo:
                return torch.as_tensor(sample.metainfo["clip_meta"], dtype=torch.float32)
        return None

    def _get_elapsed_norm(self, data_samples: SampleList, device: torch.device) -> Optional[torch.Tensor]:
        if not self.with_bdnet:
            return None
        vals = []
        scale = self.bdnet_elapsed_scale if self.bdnet_elapsed_scale > 0 else 1.0
        for sample in data_samples:
            clip_meta = self._extract_clip_meta(sample)
            if clip_meta is None:
                vals.append(torch.tensor(0.0, device=device))
                continue
            clip_meta = clip_meta.to(device=device, dtype=torch.float32).reshape(-1)
            if clip_meta.numel() >= 2:
                vals.append(clip_meta[1] / scale)
            elif clip_meta.numel() == 1:
                vals.append(clip_meta[0] / scale)
            else:
                vals.append(torch.tensor(0.0, device=device))
        return torch.stack(vals, dim=0)

    def get_backbone_features(self, inputs: torch.Tensor, elapsed_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        elif self.backbone_type == 'RSDNET':
            frame_feats = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame)  # [B, 1024]
                frame_feats.append(feats)
            latent = torch.stack(frame_feats, dim=1)  # [B, T, 1024]
        elif self.backbone_type == 'BDNET':
            frame_feats = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame, elapsed_norm=elapsed_norm)  # [B, 1664]
                frame_feats.append(feats)
            latent = torch.stack(frame_feats, dim=1)  # [B, T, 1664]
        elif self.backbone_type == 'TIMELSTM':
            frame_feats = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame)  # [B, 2048]
                frame_feats.append(feats)
            latent = torch.stack(frame_feats, dim=1)  # [B, T, 2048]
        elif self.backbone_type == 'TRANSLOCAL':
            frame_feats = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone(input_frame)  # [B, 2048]
                frame_feats.append(feats)
            latent = torch.stack(frame_feats, dim=1)  # [B, T, 2048]
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
        elapsed_norm = self._get_elapsed_norm(data_samples, device=cur_inputs.device)
        cur_vec = self.get_backbone_features(cur_inputs, elapsed_norm=elapsed_norm)

        # AVT path: z_t -> causal decoder -> z_hat_t
        if self.with_avt:
            # cur_vec: [B, T, D_enc] (e.g. D_enc=768)
            z = self.avt_feature_proj(cur_vec)  # [B, T, 2048]
            z_hat = self.decoder(z, return_logits=False, use_last_only=False)  # [B, T, 2048]

            head_input = None
            if self.head_feature_src == 'backbone': # NOTE: default consistent choice across all baselines
                head_input = cur_vec
            else:
                head_input = z_hat

            head_losses = self.cls_head.loss(head_input, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses
        
        if self.with_rulstm:
            # RU-LSTM: R-LSTM encoding + U-LSTM unrolling, then use last-step anticipated feature.
            z_hat = self.rulstm_temporal(cur_vec)  # [B, T, D]
            next_feat = self.rulstm_cls_dropout(z_hat[:, -1])  # [B, D]

            head_input = None
            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            head_losses = self.cls_head.loss(head_input, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
                
            losses.update(head_losses)
            return losses

        if self.with_slowfast_rulstm_pretrain:
            head_losses = self.cls_head.loss(cur_vec, data_samples)
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

            gate_logits = self.slowfast_fusion_mlp(torch.cat([slow_feat, fast_feat], dim=-1))  # [B, 2]
            gate = torch.softmax(gate_logits, dim=-1)
            fused_feat = gate[:, :1] * slow_feat + gate[:, 1:] * fast_feat  # [B, D]

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = fused_feat

            head_losses = self.cls_head.loss(head_input, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses

        if self.with_rsdnet:
            z_hat, _ = self.rsdnet_temporal(cur_vec)  # [B, T, D]
            z_hat = self.rsdnet_dropout(z_hat)

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            head_losses = self.cls_head.loss(head_input, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            return losses
        
        if self.with_bdnet:
            z_hat, branch_feats = self._bdnet_ensemble_forward(cur_vec)

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            head_losses = self.cls_head.loss(head_input, data_samples)
            if 'state_logits' in head_losses:
                head_losses.pop('state_logits')
            losses.update(head_losses)
            if self.bdnet_dncl_weight > 0:
                loss_div = self._bdnet_diversity_loss(branch_feats)
                if loss_div is not None:
                    losses['loss_bdnet_div'] = self.bdnet_dncl_weight * loss_div
            return losses

        if self.with_timelstm or self.with_translocal:
            head_losses = self.cls_head.loss(cur_vec, data_samples)
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

        elapsed_norm = self._get_elapsed_norm(data_samples, device=cur_inputs.device)
        cur_vec = self.get_backbone_features(cur_inputs, elapsed_norm=elapsed_norm)

        if self.with_avt:
            z = self.avt_feature_proj(cur_vec)
            z_hat = self.decoder(z, return_logits=False, use_last_only=False)
            
            head_input = None
            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            return self.cls_head.predict(head_input, data_samples)
        
        if self.with_rulstm:
            z_hat = self.rulstm_temporal(cur_vec)  # [B, T, D]
            next_feat = self.rulstm_cls_dropout(z_hat[:, -1])  # [B, D]

            head_input = None
            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            return self.cls_head.predict(head_input, data_samples)

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

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = fused_feat
            return self.cls_head.predict(head_input, data_samples)

        if self.with_rsdnet:
            z_hat, _ = self.rsdnet_temporal(cur_vec)  # [B, T, D]
            z_hat = self.rsdnet_dropout(z_hat)

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            return self.cls_head.predict(head_input, data_samples)
        
        if self.with_bdnet:
            z_hat, _ = self._bdnet_ensemble_forward(cur_vec)

            if self.head_feature_src == 'backbone':
                head_input = cur_vec
            else:
                head_input = z_hat

            return self.cls_head.predict(head_input, data_samples)

        if self.with_timelstm or self.with_translocal:
            return self.cls_head.predict(cur_vec, data_samples)

        return self.cls_head.predict(cur_vec, data_samples)
