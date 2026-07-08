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
from ..utils.ssl_model_layers import create_r50_ssl_backbone
from ..utils.misc import get_model_from_cfg
from geomloss import SamplesLoss
from mmengine.runner import load_checkpoint

@MODELS.register_module()
class StateChangeActionDistiller(Recognizer2D):
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 state_change_decoder: OptConfigType = None,
                 clustering: OptConfigType = None,
                 graph: OptConfigType = None,
                 centroid_transition: OptConfigType = None,
                 imaginator: OptConfigType = None,
                 add_futr_pred_to_centroids: bool = False,
                 use_imaginator: bool = False,
                 aux_head: OptConfigType = None,
                 fuse_centroids: OptConfigType = None,
                 freeze_backbone=False,
                 backbone_ckpt: str = '',
                 agg_feats='mean',
                 graph_on_raw_features=False,
                 use_teacher_backbone=False,
                 future_steps: int = 4,
                 teacher_affinity: str = 'cosine_sim',
                 teacher_cfg: str = 'mmaction::state_change/teacher/t_vits_videomae_verbsc_256d_k25g.py',
                 teacher_ckpt: str = 'pretrained_weights/sc_vits_cf_c25g_256to128d_cdist_flath_1441152_ep40.pth',
                 sc_emb_loss: dict = dict(type='l1_loss', loss_weight=1.0, enable=True),
                 sc_logits_loss: dict = dict(type='l1_loss', loss_weight=1.0, temp=0.3, anneal_temp=False, enable=True),
                 g_feat_loss: dict = dict(type='l1_loss', loss_weight=1.0, enable=False),
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.epoch = 0
        self.future_steps = future_steps
        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.agg_feats = agg_feats
        self.backbone_ckpt = backbone_ckpt
        self.use_imaginator = use_imaginator
        self.use_teacher_backbone = use_teacher_backbone
        self.teacher_cfg = teacher_cfg
        self.teacher_ckpt = teacher_ckpt
        self.sc_emb_loss = sc_emb_loss
        self.sc_logits_loss = sc_logits_loss
        self.g_feat_loss = g_feat_loss
        self.add_futr_pred_to_centroids = add_futr_pred_to_centroids
        self.graph_on_raw_features = graph_on_raw_features
        self.teacher_affinity = teacher_affinity

        if backbone_ckpt != '':
            self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
        
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

        self.with_centroid_transition = centroid_transition is not None
        if self.with_centroid_transition:
            self.centroid_transition = MODELS.build(centroid_transition)

        self.with_imaginator = imaginator is not None
        if self.with_imaginator:
            self.imaginator = MODELS.build(imaginator)

        self.with_aux_head = aux_head is not None
        if self.with_aux_head:
            self.aux_head = MODELS.build(aux_head)

        self.with_fuse_centroids = fuse_centroids is not None
        if self.with_fuse_centroids:
            self.fuse_centroids = MODELS.build(fuse_centroids)

        # teacher model for distillation
        self.setup_teacher()

    def setup_teacher(self):
        self.teacher = get_model_from_cfg(self.teacher_cfg)
        load_checkpoint(self.teacher, self.teacher_ckpt, strict=True)
        self.teacher._is_init = True
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        if self.use_teacher_backbone:
            self.backbone.load_state_dict(self.teacher.backbone.state_dict())

    def set_epoch(self, epoch: int, max_epochs: int) -> None:
        self.epoch = epoch

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
    
    def get_sinkhorn_based_patch_sim(self, 
                                     cur_vec: torch.Tensor, 
                                     futr_vec: torch.Tensor, 
                                     apply_sinkhorn: bool = False,
                                     enable_sparsity: bool = False) -> torch.Tensor:
        B, N, D = cur_vec.size()

        # 1. normalize the features
        cur_norm = F.normalize(cur_vec, dim=-1)
        futr_norm = F.normalize(futr_vec, dim=-1)

        # 2. get affinity matrix
        if self.teacher_affinity == 'cosine_sim':
            affinity = torch.bmm(cur_norm, futr_norm.transpose(1, 2))  # [B x THW x THW]
        else:
            affinity = -torch.cdist(cur_norm, futr_norm, p=2) # torch.matmul(feats, feats.transpose(1, 2))
            affinity = torch.exp(affinity/0.1) # lower t (0.1) values are more spread out
            affinity = affinity / affinity.sum(dim=2, keepdim=True)
        
        if apply_sinkhorn:
            # 2. Compute cost matrix based on feature similarity
            cost = 1 - affinity

            # 3. Apply Sinkhorn
            ot_solver = SamplesLoss(
                loss="sinkhorn",
                p=2,              
                blur=0.005,       # Lower blur -> more discrete matches
                scaling=0.95,     # Higher scaling -> better convergence
                reach=None,       # No reach restriction for now
                backend="tensorized",
                debias=False,
                potentials=True
            )
            
            # Process each batch
            affinity = torch.zeros_like(cost)
            
            for b in range(B):
                # Prepare weights (uniform distribution)
                weights_src = torch.ones(N, device=cur_vec.device) / N
                weights_tgt = torch.ones(N, device=cur_vec.device) / N
                
                # Get potentials
                f, g = ot_solver(
                    weights_src, cur_norm[b],     # α, x (source)
                    weights_tgt, futr_norm[b]      # β, y (target)
                )
                
                # Convert potentials to affinity
                f_expand = f.view(-1, 1)   # [THW x 1]
                g_expand = g.view(1, -1)   # [1 x THW]
                
                # Compute affinity (negative cost with potentials)
                aff = -cost[b] - (f_expand + g_expand)  # [THW x THW]
                affinity[b] = F.softmax(aff / 0.1, dim=-1)
        
        # 4. Ensure sparsity with top-k
        if enable_sparsity:
            topk = min(5, N)  # Keep top-5 matches per patch
            values, _ = torch.topk(affinity, k=topk, dim=-1)
            mask = affinity >= values[..., -1:]
            affinity = affinity * mask.float()

        return affinity
    
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
    
    def get_teacher_outputs(self, cur_inputs, futr_inputs) -> dict:
        b, t, h, w, _ = cur_inputs.size()
        with torch.no_grad():
            t_futr_vec, t_futr_last_tokens = self.get_backbone_features(futr_inputs)
            t_cur_vec, t_cur_last_tokens = self.get_backbone_features(cur_inputs)
        
            t_cur_vec = t_cur_vec.flatten(1, 3)
            t_futr_vec = t_futr_vec.flatten(1, 3)

            if self.graph_on_raw_features:
                t_g_cur = self.teacher.cluster_graph.create_graph(t_cur_vec)
                t_g_futr = self.teacher.cluster_graph.create_graph(t_futr_vec)
                t_cur_vec = self.teacher.cluster_graph.message_passing(t_g_cur, t_cur_vec)
                t_futr_vec = self.teacher.cluster_graph.message_passing(t_g_futr, t_futr_vec)

            # clustering to get centroids NOTE: !!! teacher not used in previous codebase
            if self.with_sinkhorn_clustering:
                t_cur_centroids, t_cur_asm = self.teacher.sinkhorn_clustering(t_cur_vec, t_cur_last_tokens)
                t_futr_centroids, t_futr_asm = self.teacher.sinkhorn_clustering(t_futr_vec, t_futr_last_tokens)
            else:
                t_cur_centroids = t_cur_vec
                t_futr_centroids = t_futr_vec

            patch_affinity = self.get_sinkhorn_based_patch_sim(t_cur_vec, t_futr_vec, apply_sinkhorn=False, enable_sparsity=False)

            # graph processing from teacher model
            if self.with_cluster_graph and not self.graph_on_raw_features:
                t_g_cur = self.teacher.cluster_graph.create_graph(t_cur_centroids)
                m_cur_centroids = self.teacher.cluster_graph.message_passing(t_g_cur, t_cur_centroids)
                t_g_futr = self.teacher.cluster_graph.create_graph(t_futr_centroids)
                m_futr_centroids = self.teacher.cluster_graph.message_passing(t_g_futr, t_futr_centroids)
                centroids_t = torch.stack([m_cur_centroids, m_futr_centroids], dim=1)
            else:
                centroids_t = torch.stack([t_cur_centroids, t_futr_centroids], dim=1)
                
            # centroids_t = centroids_t.mean(dim=1) if self.agg_feats == 'mean' else centroids_t.sum(dim=1)
            centroids_t = self.aggregate_features(centroids_t)

            if not self.with_sinkhorn_clustering:
                centroids_t = centroids_t.view(b, t, h, w, -1).mean(dim=[2, 3])

            if self.with_state_change_decoder:
                cls_embs_t = self.teacher.state_change_decoder(centroids_t)
            else:
                cls_embs_t = centroids_t

            # logits from the head of teacher model
            t_sc_logits = self.teacher.cls_head(cls_embs_t)

        return dict(
            t_cur_vec=t_cur_vec,
            t_futr_vec=t_futr_vec,
            t_cur_centroids=t_cur_centroids,
            t_futr_centroids=t_futr_centroids,
            cls_embs_t=cls_embs_t,
            t_sc_logits=t_sc_logits,
            t_cur_asm=t_cur_asm,
            t_futr_asm=t_futr_asm,
            patch_affinity=patch_affinity)

    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        cur_inputs, futr_inputs = torch.split(inputs, 1, dim=1)
        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        futr_inputs = futr_inputs.view((-1, ) + futr_inputs.shape[2:])

        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3)
        losses = dict()

        # get teacher outputs
        t_outputs = self.get_teacher_outputs(cur_inputs, futr_inputs)
        t_cur_vec, t_futr_vec = t_outputs['t_cur_vec'], t_outputs['t_futr_vec']
        t_cur_centroids, t_futr_centroids = t_outputs['t_cur_centroids'], t_outputs['t_futr_centroids']
        cls_embs_t, t_sc_logits = t_outputs['cls_embs_t'], t_outputs['t_sc_logits']
        t_cur_asm, t_futr_asm = t_outputs['t_cur_asm'], t_outputs['t_futr_asm']
        patch_affinity = t_outputs['patch_affinity']

        # auxillary future prediction loss
        if self.with_imaginator:
            t_futr_vec = rearrange(t_futr_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            cur_vec_pooled = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            futr_pred_latent = self.imaginator(cur_vec_pooled, future_steps=self.future_steps)
            aux_loss = self.aux_head.loss(futr_pred_latent, t_futr_vec)
            losses.update(aux_loss)

        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cur_centroids, cur_assignment = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
        else:
            cur_centroids = cur_vec

        # compute centroid transition before graph processing NOTE: alternative is to transition after graph processing
        loss_trans, transition_matrix, futr_centroids = self.centroid_transition.loss(cur_centroids, 
                                                                                    t_futr_asm,
                                                                                    t_futr_centroids,
                                                                                    self.epoch,
                                                                                    t_cur_centroids,
                                                                                    t_cur_asm,
                                                                                    patch_affinity)

        losses.update(loss_trans) # NOTE: not used in previous codebases

        # graph processing
        if self.with_cluster_graph:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            m_cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

            g_futr = self.cluster_graph.create_graph(futr_centroids)
            m_futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)
        else:
            m_cur_centroids = cur_centroids
            m_futr_centroids = futr_centroids

        # if self.g_feat_loss['enable']:
        #     loss_g_feat = self.g_feat_loss['loss_weight'] * getattr(F, self.g_feat_loss['type'])(m_cur_centroids, t_g_futr_centroids.detach())
        #     losses['loss_g_feat'] = loss_g_feat

        # stack cur and predicted future centroids
        centroids = torch.stack([m_cur_centroids, m_futr_centroids], dim=1)
        # centroids = centroids.mean(dim=1) if self.agg_feats == 'mean' else centroids.sum(dim=1)
        centroids = self.aggregate_features(centroids)

        if not self.with_sinkhorn_clustering:
            centroids = centroids.view(b, t, h, w, -1).mean(dim=[2,3])

        if self.with_fuse_centroids:
            centroids = self.fuse_centroids(futr_pred_latent, centroids)

        # state representation to state change representation
        if self.with_state_change_decoder:
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids

        if self.sc_emb_loss['enable']:
            loss_sc_emb = self.sc_emb_loss['loss_weight'] * getattr(F, self.sc_emb_loss['type'])(state_change_class_embs, cls_embs_t.detach())
            losses['loss_sc_emb'] = loss_sc_emb

        loss_sc = self.cls_head.loss(state_change_class_embs, data_samples)
        if 'state_logits' in loss_sc:
            sc_logit = loss_sc.pop('state_logits')
        losses.update(loss_sc)

        if self.sc_logits_loss['enable']:
            if self.sc_logits_loss['anneal_temp']:
                temp = max(0.5, min(1.0, math.exp(-0.05 * (self.epoch - 5))))
            else:
                temp = self.sc_logits_loss['temp']

            s_probs = F.softmax(sc_logit/temp, dim=-1)
            t_sc_probs = F.softmax(t_sc_logits.detach()/temp, dim=-1)

            loss_sc_logits = self.sc_logits_loss['loss_weight'] * getattr(F, self.sc_logits_loss['type'])(s_probs, t_sc_probs)
            losses['loss_sc_logits'] = loss_sc_logits

        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        cur_inputs, futr_inputs = torch.split(inputs, 1, dim=1)
        cur_inputs = cur_inputs.view((-1, ) + cur_inputs.shape[2:])
        futr_inputs = futr_inputs.view((-1, ) + futr_inputs.shape[2:])

        cur_vec, cur_last_tokens  = self.get_backbone_features(cur_inputs)
        b, t, h, w, d = cur_vec.size()
        cur_vec = cur_vec.flatten(1, 3)

        # auxillary future prediction loss
        if self.with_imaginator:
            cur_vec_pooled = rearrange(cur_vec, 'b (t h w) d -> b t h w d', t=t, h=h, w=w).mean(dim=[2, 3])
            futr_pred_latent = self.imaginator(cur_vec_pooled, future_steps=self.future_steps)

        # clustering to get centroids
        if self.with_sinkhorn_clustering:
            cur_centroids, cur_assignment = self.sinkhorn_clustering(cur_vec, cur_last_tokens)
        else:
            cur_centroids = cur_vec

        # visualize_clustering(cur_inputs, cur_assignment, 3, ignore_even_frames=True)
        transition_matrix, futr_centroids = self.centroid_transition.predict(cur_centroids)

        # graph processing
        if self.with_cluster_graph:
            g_cur = self.cluster_graph.create_graph(cur_centroids)
            m_cur_centroids = self.cluster_graph.message_passing(g_cur, cur_centroids)

            g_futr = self.cluster_graph.create_graph(futr_centroids)
            m_futr_centroids = self.cluster_graph.message_passing(g_futr, futr_centroids)
        else:
            m_cur_centroids = cur_centroids
            m_futr_centroids = futr_centroids

        # stack cur and predicted future centroids
        centroids = torch.stack([m_cur_centroids, m_futr_centroids], dim=1)
        # centroids = centroids.mean(dim=1) if self.agg_feats == 'mean' else centroids.sum(dim=1)
        centroids = self.aggregate_features(centroids)

        if not self.with_sinkhorn_clustering:
            centroids = centroids.view(b, t, h, w, -1).mean(dim=[2, 3])

        if self.with_fuse_centroids:
            centroids = self.fuse_centroids(futr_pred_latent, centroids)

        # state representation to state change representation
        if self.with_state_change_decoder:
            state_change_class_embs = self.state_change_decoder(centroids)
        else:
            state_change_class_embs = centroids
        
        return self.cls_head.predict(state_change_class_embs, data_samples)
