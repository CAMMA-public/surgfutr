"""
Spatio-temporal cluster graph on current and future clips acting as a teacher model.
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
    create_r50_ssl_backbone, create_endovit_backbone, create_surgenet_backbone)

@MODELS.register_module()
class VideoClusterGraphWithFutr(Recognizer2D):
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 state_change_decoder: OptConfigType = None,
                 clustering: OptConfigType = None,
                 graph: OptConfigType = None,
                 imaginator: OptConfigType = None,
                 centroid_transition: OptConfigType = None,
                 aux_head: OptConfigType = None,
                 use_imaginator=False,
                 future_steps: int = 4,
                 freeze_backbone=False,
                 graph_on_raw_features=False,
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
        self.graph_on_raw_features = graph_on_raw_features

        if backbone_ckpt != '' and backbone_type == 'ResNet50_SSL':
            self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        elif backbone_ckpt != '' and backbone_type == 'EndoViT':
            self.backbone = create_endovit_backbone(backbone_ckpt, freeze_backbone)
        elif backbone_ckpt != '' and backbone_type == 'SurgeNet':
            self.backbone = create_surgenet_backbone(backbone_ckpt, freeze_backbone)

        self.with_state_change_decoder = state_change_decoder is not None
        if self.with_state_change_decoder:
            self.state_change_decoder = MODELS.build(state_change_decoder)

        if self.freeze_backbone:
            self.backbone.requires_grad = False
            # self.backbone.patch_embed.requires_grad = False # TODO: turn off during training ?
            print("Freezing backbone")

        self.with_sinkhorn_clustering = clustering is not None
        if self.with_sinkhorn_clustering:
            self.sinkhorn_clustering = MODELS.build(clustering)

        self.with_cluster_graph = graph is not None
        if self.with_cluster_graph:
            self.cluster_graph = MODELS.build(graph)

        self.with_imaginator = imaginator is not None
        if self.with_imaginator:
            self.imaginator = MODELS.build(imaginator)

        self.with_centroid_transition = centroid_transition is not None
        if self.with_centroid_transition:
            self.centroid_transition = MODELS.build(centroid_transition)

    def preprocess_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == 'MViT':
            patch_tokens, cls_token, last_tokens = inputs[0], inputs[1], inputs[-1]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type in ['VisionTransformer', 'InternVideo2', 'CustomSwin', 'CustomUniFormerV2']:
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
            last_tokens = None
        elif self.backbone_type == 'EndoViT':
            b, c, t, h, w = inputs.shape
            latent = []
            for i in range(t):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone.forward_features(input_frame) # b x (h x w + 1) x 768
                # remove cls token
                feats = feats[:, 1:, :]
                # reshape to b x c x h x w
                feats = rearrange(feats, 'b (h w) c -> b c h w', h=14, w=14) # fixed patch size of 16
                latent.append(feats)
            latent = torch.stack(latent, dim=2)
            latent = rearrange(latent, 'b c t h w -> b t h w c')
            b, t, h, w, c = latent.size()
            last_tokens = torch.empty(b, 1, t*h*w, t*h*w).to(inputs.device)
        elif self.backbone_type == 'SurgeNet':
            b, c, t, h, w = inputs.shape
            latent = []
            for i in range(t):
                input_frame = inputs[:, :, i, :, :]
                feats = self.backbone.forward_features(input_frame) # b x (h x w + 1) x 768
                feats = feats[1][3]
                
                # reshape to b x c x h x w
                # feats = rearrange(feats, 'b (h w) c -> b c h w', h=7, w=7) # fixed patch size of 16
                latent.append(feats)
            latent = torch.stack(latent, dim=2)
            latent = rearrange(latent, 'b c t h w -> b t h w c')
            b, t, h, w, c = latent.size()
            last_tokens = torch.empty(b, 1, t*h*w, t*h*w).to(inputs.device)
        else:
            features = self.backbone(inputs)
            latent, last_tokens = self.preprocess_features(features)
            
        if self.with_neck:
            latent = self.neck(latent)
        return latent, last_tokens

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
         
        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        
        cur_vec = cur_vec.flatten(1, 3).contiguous()
        losses = dict()
        
        if self.graph_on_raw_features:
            g_cur = self.cluster_graph.create_graph(cur_vec)
            cur_vec = self.cluster_graph.message_passing(g_cur, cur_vec)

        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cluster_out = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
            if len(cluster_out) == 3:
                cur_centroids, cur_asm, cur_asm_norm = cluster_out
            else:
                cur_centroids, cur_asm = cluster_out
        else:
            cur_centroids = cur_vec

        # graph processing
        if self.with_cluster_graph and not self.graph_on_raw_features:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

        if not self.with_sinkhorn_clustering:
            cur_centroids = cur_centroids.view(b, t, h, w, -1)
            cur_centroids = cur_centroids.mean(dim=[2, 3]) # using time dimension only. check with thw dimension.
        
        if self.use_futr:
            futr_vec, futr_last_tokens = self.get_backbone_features(futr_inputs)
            futr_vec = futr_vec.flatten(1, 3)

            if self.graph_on_raw_features:
                g_futr = self.cluster_graph.create_graph(futr_vec)
                futr_vec = self.cluster_graph.message_passing(g_futr, futr_vec)

            if self.with_sinkhorn_clustering:
                cluster_out = self.sinkhorn_clustering(futr_vec, futr_last_tokens)
                if len(cluster_out) == 3:
                    futr_centroids, futr_asm, futr_asm_norm = cluster_out
                else:
                    futr_centroids, futr_asm = cluster_out
            else:
                futr_centroids = futr_vec

            # graph processing NOTE: shared graph for both current and future?
            if self.with_cluster_graph and not self.graph_on_raw_features:
                g_futr = self.cluster_graph.create_graph(futr_centroids)
                futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)

            if not self.with_sinkhorn_clustering:
                futr_centroids = futr_centroids.view(b, t, h, w, -1)
                futr_centroids = futr_centroids.mean(dim=[2, 3])

            centroids = torch.stack([cur_centroids, futr_centroids], dim=1) # aggregate state representations from current and future
        else:
            centroids = cur_centroids

        centroids = self.aggregate_features(centroids)

        # imaginator if enabled
        if self.with_imaginator:
            future_pred_vec = self.imaginator(centroids, self.future_steps) # still the current centroids
            centroids = torch.stack([centroids, future_pred_vec], dim=1).mean(dim=1)
        
        # state representation to state change representation
        if self.with_state_change_decoder:
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids

        loss_head = self.cls_head.loss(state_change_class_embs, data_samples)
        if 'state_logits' in loss_head:
            sc_logit = loss_head.pop('state_logits')
        losses.update(loss_head)

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

        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3).contiguous()

        if self.graph_on_raw_features:
            g_cur = self.cluster_graph.create_graph(cur_vec)
            cur_vec = self.cluster_graph.message_passing(g_cur, cur_vec)

        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cluster_out = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
            if len(cluster_out) == 3:
                cur_centroids, cur_asm, cur_asm_norm = cluster_out
            else:
                cur_centroids, cur_asm = cluster_out
        else:
            cur_centroids = cur_vec

        # graph processing
        if self.with_cluster_graph and not self.graph_on_raw_features:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

        if not self.with_sinkhorn_clustering:
            cur_centroids = cur_centroids.view(b, t, h, w, -1)
            cur_centroids = cur_centroids.mean(dim=[2, 3])
        
        if self.use_futr:
            futr_vec, futr_last_tokens = self.get_backbone_features(futr_inputs)
            futr_vec = futr_vec.flatten(1, 3)

            if self.graph_on_raw_features:
                g_futr = self.cluster_graph.create_graph(futr_vec)
                futr_vec = self.cluster_graph.message_passing(g_futr, futr_vec)

            if self.with_sinkhorn_clustering:
                cluster_out = self.sinkhorn_clustering(futr_vec, futr_last_tokens)
                if len(cluster_out) == 3:
                    futr_centroids, futr_asm, futr_asm_norm = cluster_out
                else:
                    futr_centroids, futr_asm = cluster_out
            else:
                futr_centroids = futr_vec

            # graph processing
            if self.with_cluster_graph and not self.graph_on_raw_features:
                g_futr = self.cluster_graph.create_graph(futr_centroids)
                futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)

            if not self.with_sinkhorn_clustering:
                futr_centroids = futr_centroids.view(b, t, h, w, -1)
                futr_centroids = futr_centroids.mean(dim=[2, 3])

            # aggregate state representations from current and future
            centroids = torch.stack([cur_centroids, futr_centroids], dim=1)
        else:
            centroids = cur_centroids

        centroids = self.aggregate_features(centroids)

        # imaginator if enabled
        if self.with_imaginator:
            future_pred_vec = self.imaginator(centroids, self.future_steps) # still the current centroids
            centroids = torch.stack([centroids, future_pred_vec], dim=1).mean(dim=1)

        # state representation to state change representation
        if self.with_state_change_decoder:
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids
            
        return self.cls_head.predict(state_change_class_embs, data_samples)
