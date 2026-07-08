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
from ..utils.ssl_model_layers import create_r50_ssl_backbone

@MODELS.register_module()
class SCADynDownstreamRecognizer(Recognizer2D):
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 state_change_decoder: OptConfigType = None,
                 clustering: OptConfigType = None,
                 graph: OptConfigType = None,
                 imaginator: OptConfigType = None,
                 centroid_transition: OptConfigType = None,
                 use_centroid_transition=False,
                 merge_all_features=False,
                 condition_on_global_feats=False,
                 future_steps: int = 4,
                 freeze_backbone=False,
                 backbone_ckpt: str = '',
                 agg_feats='mean',
                 use_futr=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.epoch = 0
        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.use_futr = use_futr
        self.agg_feats = agg_feats
        self.backbone_ckpt = backbone_ckpt
        self.future_steps = future_steps
        self.merge_all_features = merge_all_features
        self.condition_on_global_feats = condition_on_global_feats

        if backbone_ckpt != '':
            self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)

        self.with_state_change_decoder = state_change_decoder is not None
        if self.with_state_change_decoder:
            self.state_change_decoder = MODELS.build(state_change_decoder)

        self.with_centroid_transition = centroid_transition is not None
        if self.with_centroid_transition:
            self.centroid_transition = MODELS.build(centroid_transition)

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

    def preprocess_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == 'MViT':
            patch_tokens, cls_token, last_tokens = inputs[0], inputs[1], inputs[-1]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type == 'VisionTransformer':
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

        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3).contiguous() # downstream eval stage - A
        losses = dict()

        # condition on global features
        cur_vec_global = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[1, 2, 3])

        # imaginator if enabled -- downstream eval stage - B
        if self.with_imaginator:
            cur_vec_pooled = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            future_pred_vec = self.imaginator(cur_vec_pooled, future_steps=self.future_steps)
            if not self.merge_all_features:
                cur_vec = torch.stack([cur_vec_pooled, future_pred_vec], dim=1) # b x t x t x d 
        
        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cur_centroids, cur_assignment = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
        else:
            cur_centroids = cur_vec

        if self.with_centroid_transition: # downstream eval stage - C
            if self.condition_on_global_feats:
                transition_matrix, futr_centroids = self.centroid_transition.predict(cur_vec_global, cur_centroids)
            else:
                transition_matrix, futr_centroids = self.centroid_transition.predict(cur_centroids)

        # graph processing
        if self.with_cluster_graph:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

        if not self.with_sinkhorn_clustering and not self.with_imaginator:
            cur_centroids = cur_centroids.view(b, t, h, w, -1)
            cur_centroids = cur_centroids.mean(dim=[2, 3]) # using time dimension only. check with thw dimension.

        if self.with_centroid_transition: # downstream eval stage - C
            if self.with_cluster_graph:
                g_futr = self.cluster_graph.create_graph(futr_centroids)
                m_futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)
                centroids = torch.stack([cur_centroids, m_futr_centroids], dim=1)
            else:
                centroids = torch.stack([cur_centroids, futr_centroids], dim=1)
        else:
            centroids = cur_centroids
        
        # state representation to state change representation
        if self.with_state_change_decoder: # downstream eval stage - D
            if self.merge_all_features:
                centroids_before_sc = centroids
            # centroids = centroids.mean(dim=1) if self.agg_feats == 'mean' else centroids.sum(dim=1)
            centroids = self.aggregate_features(centroids)
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids

        if not self.merge_all_features:
            loss_head = self.cls_head.loss(state_change_class_embs, data_samples)
        else:
            loss_head = self.cls_head.loss(
                state_change_class_embs,
                centroids_before_sc, 
                cur_vec_pooled,
                future_pred_vec,
                data_samples)

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
        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3).contiguous() # downstream eval stage - A

        # condition on global features
        cur_vec_global = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[1, 2, 3])

        # imaginator if enabled -- downstream eval stage - B
        if self.with_imaginator:
            cur_vec_pooled = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            future_pred_vec = self.imaginator(cur_vec_pooled, future_steps=self.future_steps)
            if not self.merge_all_features:
                cur_vec = torch.stack([cur_vec_pooled, future_pred_vec], dim=1) # 

        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cur_centroids, cur_assignment = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
        else:
            cur_centroids = cur_vec
        
        if self.with_centroid_transition: # downstream eval stage - C
            if self.condition_on_global_feats:
                transition_matrix, futr_centroids = self.centroid_transition.predict(cur_vec_global, cur_centroids)
            else:
                transition_matrix, futr_centroids = self.centroid_transition.predict(cur_centroids)

        # graph processing
        if self.with_cluster_graph:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

        if not self.with_sinkhorn_clustering and not self.with_imaginator:
            cur_centroids = cur_centroids.view(b, t, h, w, -1)
            cur_centroids = cur_centroids.mean(dim=[2, 3])

        if self.with_centroid_transition: # downstream eval stage - C
            if self.with_cluster_graph:
                g_futr = self.cluster_graph.create_graph(futr_centroids)
                m_futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)
                centroids = torch.stack([cur_centroids, m_futr_centroids], dim=1)
            else:
                centroids = torch.stack([cur_centroids, futr_centroids], dim=1)
        else:
            centroids = cur_centroids

        # state representation to state change representation
        if self.with_state_change_decoder: # downstream eval stage - D
            if self.merge_all_features:
                centroids_before_sc = centroids
            # centroids = centroids.mean(dim=1) if self.agg_feats == 'mean' else centroids.sum(dim=1)
            centroids = self.aggregate_features(centroids)
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids
        
        if not self.merge_all_features:
            return self.cls_head.predict(state_change_class_embs, data_samples)
        else:
            return self.cls_head.predict(
                state_change_class_embs,
                centroids_before_sc, 
                cur_vec_pooled,
                future_pred_vec,
                data_samples)
