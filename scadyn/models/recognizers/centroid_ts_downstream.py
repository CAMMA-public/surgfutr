"""
SCADyn: State Change Action Dynamics for Surgical Future Prediction
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from mmaction.models.recognizers import Recognizer2D
from mmaction.utils import (OptConfigType,SampleList)
from mmaction.registry import MODELS
from einops import rearrange
from mmpretrain.models.utils import CosineEMA
from ..utils.ssl_model_layers import create_r50_ssl_backbone, create_endovit_backbone, create_surgenet_backbone
from ..utils.misc import get_model_from_cfg
from mmengine.runner import load_checkpoint
from geomloss import SamplesLoss

@MODELS.register_module()
class CentroidTSDownstream(Recognizer2D):
    """
    Teacher with future clip features and centroids. Two ways to use teacher:
    1. Student predicts either the spatial pooled future clip features or the centroids
    2. Student predicts the transition from current centroids to future centroids and the current and future centroids are 
       aggregated to get the state change representation
    """
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 state_change_decoder: OptConfigType = None,
                 clustering: OptConfigType = None,
                 graph: OptConfigType = None,
                 centroid_transition: OptConfigType = None,
                 aux_head: OptConfigType = None,
                 imaginator: OptConfigType = None,
                 freeze_backbone=False,
                 backbone_ckpt: str = '',
                 agg_feats='mean',
                 use_futr: bool = False,
                 num_classes: int = 10,
                 num_states: int = 4,
                 future_steps: int = 4,
                 teacher_cfg: str = '',
                 teacher_ckpt: str = '',
                 disable_backbone_spatial_pooling: bool = False,
                 combine_backbone_centroids: bool = False,
                 use_teacher_backbone: bool = False,
                 use_normalized_assignment: bool = False,
                 use_centroid_transition: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.epoch = 0
        self.future_steps = future_steps
        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.agg_feats = agg_feats
        self.backbone_ckpt = backbone_ckpt
        self.num_classes = num_classes
        self.num_states = num_states
        self.teacher_cfg = teacher_cfg
        self.teacher_ckpt = teacher_ckpt
        self.use_teacher_backbone = use_teacher_backbone
        self.use_normalized_assignment = use_normalized_assignment
        self.use_centroid_transition = use_centroid_transition
        self.disable_backbone_spatial_pooling = disable_backbone_spatial_pooling
        self.combine_backbone_centroids = combine_backbone_centroids

        # if backbone_ckpt != '':
        #     self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        # if backbone_ckpt != '' and backbone_type == 'ResNet50_SSL':
        #     self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        # elif backbone_ckpt != '' and backbone_type == 'EndoViT':
        #     self.backbone = create_endovit_backbone(backbone_ckpt, freeze_backbone)

        if backbone_ckpt != '' and backbone_type == 'ResNet50_SSL':
            self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        elif backbone_ckpt != '' and backbone_type == 'EndoViT':
            self.backbone = create_endovit_backbone(backbone_ckpt, freeze_backbone)
        elif backbone_ckpt != '' and backbone_type == 'SurgeNet':
            self.backbone = create_surgenet_backbone(backbone_ckpt, freeze_backbone)
        elif backbone_type == 'InternVideo2':
            raise NotImplementedError(
                'CentroidTSDownstream requests InternVideo2, but '
                '`create_internvideo2_backbone` is no longer defined in '
                '`scadyn.models.utils.ssl_model_layers`.')
        
        if self.freeze_backbone:
            self.backbone.requires_grad = False
            print("Freezing backbone")

        self.with_clustering = clustering is not None
        if self.with_clustering:
            self.clustering = MODELS.build(clustering)

        self.with_cluster_graph = graph is not None
        if self.with_cluster_graph:
            self.cluster_graph = MODELS.build(graph)

        self.with_centroid_transition = centroid_transition is not None
        if self.with_centroid_transition:
            self.centroid_transition = MODELS.build(centroid_transition)

        self.with_imaginator = imaginator is not None
        if self.with_imaginator:
            self.imaginator = MODELS.build(imaginator)

        self.with_state_change_decoder = state_change_decoder is not None
        if self.with_state_change_decoder:
            self.state_change_decoder = MODELS.build(state_change_decoder)

        self.with_aux_head = aux_head is not None
        if self.with_aux_head:
            self.aux_head = MODELS.build(aux_head)

    def set_epoch(self, epoch: int, max_epochs: int) -> None:
        self.epoch = epoch

    def preprocess_features(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backbone_type == 'MViT':
            patch_tokens, cls_token, last_tokens = inputs[0], inputs[1], inputs[-1]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type == 'VisionTransformer' or self.backbone_type == 'InternVideo2' or self.backbone_type == 'CustomSwin':
            patch_tokens, last_tokens = inputs
        return patch_tokens, last_tokens
    
    def get_backbone_features(self, inputs: torch.Tensor, backbone: torch.nn.Module=None) -> torch.Tensor:
        backbone = backbone if backbone is not None else self.backbone
        if self.backbone_type == 'ResNet50_SSL':
            stacked_feats_2d = []
            for i in range(inputs.shape[2]):
                input_frame = inputs[:, :, i, :, :]
                feats = backbone(input_frame) # b x c x h x w
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
            features = backbone(inputs)
            latent, last_tokens = self.preprocess_features(features)

        if self.with_neck:
            latent = self.neck(latent)
        return latent, last_tokens
    
    def aggregate_features(self, centroids: torch.Tensor) -> torch.Tensor:
        if self.agg_feats == 'mean':
            centroids = centroids.mean(dim=1)
        elif self.agg_feats == 'sum':
            centroids = centroids.sum(dim=1)
        elif self.agg_feats == 'flatten':
            centroids = centroids.flatten(1, 2) # (2 x K) x d
        else:
            centroids = centroids
        return centroids

    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        if inputs.size(1) == 2:
            cur_inputs, futr_inputs = torch.split(inputs, 1, dim=1)
        else:
            cur_inputs = inputs
            futr_inputs = inputs.clone()
            
        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        futr_inputs = futr_inputs.view((-1, ) + futr_inputs.shape[2:])

        if self.freeze_backbone:
            with torch.no_grad():
                cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        else:
            cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)

        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3)

        # clustering
        if self.with_clustering:    
            cur_centroids, cur_asm, cur_asm_norm = self.clustering(cur_vec, cur_last_tokens)
            if self.use_normalized_assignment:
                cur_asm = cur_asm_norm
        else:
            cur_vec = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w)
            if not self.disable_backbone_spatial_pooling:
                cur_vec = cur_vec.mean(dim=[2, 3])

        losses = dict()

        # assignment offset eval mode
        if self.with_centroid_transition:
            with torch.no_grad():
                assignment_offset_matrix, pred_futr_centroids = self.centroid_transition.predict(cur_centroids) #, cur_asm, cur_vec)
            # stack cur and predicted future centroids
            centroids = torch.stack([cur_centroids, pred_futr_centroids], dim=1)
            centroids = self.aggregate_features(centroids)
        else:
            centroids = cur_vec

        # TODO: optional graph encoding from the centroids

        # auxillary future prediction on spatial pooled features (for SurgFutr)
        if self.with_imaginator:
            futr_pred_latent = self.imaginator(cur_vec, future_steps=self.future_steps)
            centroids = torch.stack([cur_vec, futr_pred_latent], dim=1).flatten(1, 2)

        if self.combine_backbone_centroids:
            cur_vec = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            centroids = torch.cat([cur_vec, centroids], dim=1) # b x (t + k) x d

        loss_cls = self.cls_head.loss(centroids, data_samples)

        losses.update(loss_cls)

        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
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
        cur_vec = cur_vec.flatten(1, 3)

        if self.with_clustering:
            cur_centroids, cur_asm, cur_asm_norm = self.clustering(cur_vec, cur_last_tokens)
            if self.use_normalized_assignment:
                cur_asm = cur_asm_norm
        else:
            cur_vec = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w)
            if not self.disable_backbone_spatial_pooling:
                cur_vec = cur_vec.mean(dim=[2, 3])

        # predict offset in the assignment matrix
        if self.with_centroid_transition:
            assignment_offset_matrix, pred_futr_centroids = self.centroid_transition.predict(cur_centroids) #, cur_asm, cur_vec)
            # stack cur and predicted future centroids
            centroids = torch.stack([cur_centroids, pred_futr_centroids], dim=1)
            centroids = self.aggregate_features(centroids)
        else:
            centroids = cur_vec

        # auxillary future prediction loss
        if self.with_imaginator:
            futr_pred_latent = self.imaginator(cur_vec, future_steps=self.future_steps)
            centroids = torch.stack([cur_vec, futr_pred_latent], dim=1).flatten(1, 2)

        if self.combine_backbone_centroids:
            cur_vec = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            centroids = torch.cat([cur_vec, centroids], dim=1) # b x (t + k) x d

        # TODO:visualize_clustering(cur_inputs, cur_assignment, 3, ignore_even_frames=True)

        return self.cls_head.predict(centroids, data_samples)
