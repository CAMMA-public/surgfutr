import torch
import torch.nn.functional as F
from mmaction.registry import MODELS
from mmaction.utils import (OptConfigType, SampleList)
from mmaction.models.recognizers import Recognizer2D
from mmpretrain.models.utils import CosineEMA
from einops import rearrange
from ..utils.ssl_model_layers import (create_r50_ssl_backbone, create_endovit_backbone,
                            create_surgenet_backbone)

@MODELS.register_module()
class SurgFutrPretrainer(Recognizer2D):
    """ 
    SurgFUTR pretraining model by autoregressive prediction of 
    spatially pooled future clip features 
    """
    def __init__(self,
                 backbone_type: str = 'VisionTransformer',
                 imaginator: OptConfigType = None,
                 future_steps: int = 1,
                 base_momentum: float = 0.004,
                 state_change_decoder: OptConfigType = None,
                 backbone_ckpt: str = '',
                 freeze_backbone: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.epoch = 0
        self.future_steps = future_steps
        self.backbone_type = backbone_type
        self.imaginator = MODELS.build(imaginator)

        self.with_state_change_decoder = state_change_decoder is not None
        if self.with_state_change_decoder:
            self.state_change_decoder = MODELS.build(state_change_decoder)

        if backbone_ckpt != '':
            if backbone_type == 'ResNet50_SSL':
                self.backbone = create_r50_ssl_backbone(self.backbone, backbone_ckpt, freeze_backbone)
            elif backbone_type == 'EndoViT':
                self.backbone = create_endovit_backbone(backbone_ckpt, freeze_backbone)
            elif backbone_type == 'SurgeNet':
                self.backbone = create_surgenet_backbone(backbone_ckpt, freeze_backbone)
        
        if not self.with_neck:
            self.neck = torch.nn.Identity()
        self.target_net = CosineEMA(
            torch.nn.Sequential(self.backbone, self.neck), momentum=base_momentum)

    def preprocess_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == 'MViT':
            patch_tokens, last_tokens, attn_matrix = inputs[0], inputs[1], inputs[2]
            patch_tokens = rearrange(patch_tokens, 'b d t h w -> b t h w d')
        elif self.backbone_type == 'VisionTransformer':
            patch_tokens, last_tokens = inputs
        return patch_tokens, last_tokens
    
    def get_r50_ssl_features(self, inputs: torch.Tensor, feature_extractor) -> torch.Tensor:
        stacked_feats_2d = []
        for i in range(inputs.shape[2]):
            input_frame = inputs[:, :, i, :, :]
            feats = feature_extractor(input_frame) # b x c x h x w
            stacked_feats_2d.append(feats)
        stacked_feats_2d = torch.stack(stacked_feats_2d, dim=2)
        features = rearrange(stacked_feats_2d, 'b c t h w -> b t h w c')
        return features

    def get_endovit_ssl_features(self, inputs: torch.Tensor, feature_extractor) -> torch.Tensor:        
        b, c, t, h, w = inputs.shape
        stacked_feats_2d = []
        for i in range(t):
            input_frame = inputs[:, :, i, :, :]
            feats = feature_extractor.forward_features(input_frame) # b x (h x w + 1) x 768
            # feats = feats[1][3]
            feats = feats[:, 1:, :]
            
            # reshape to b x c x h x w
            # feats = rearrange(feats, 'b (h w) c -> b c h w', h=7, w=7) # fixed patch size of 16
            stacked_feats_2d.append(feats)
        stacked_feats_2d = torch.stack(stacked_feats_2d, dim=1)
        features = rearrange(stacked_feats_2d, 'b t (h w) c -> b t h w c', h=14, w=14)
        return features

    def get_surgenet_ssl_features(self, inputs: torch.Tensor, feature_extractor) -> torch.Tensor:
        b, c, t, h, w = inputs.shape
        stacked_feats_2d = []
        for i in range(t):
            input_frame = inputs[:, :, i, :, :]
            feats = self.backbone.forward_features(input_frame) # b x (h x w + 1) x 768
            feats = feats[1][3]
            # reshape to b x c x h x w
            # feats = rearrange(feats, 'b (h w) c -> b c h w', h=7, w=7) # fixed patch size of 16
            stacked_feats_2d.append(feats)
        stacked_feats_2d = torch.stack(stacked_feats_2d, dim=2)
        features = rearrange(stacked_feats_2d, 'b c t h w -> b t h w c')
        return features
    
    def process_future_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        target_backbone = self.target_net.module[0]
        target_neck     = self.target_net.module[1]

        if self.backbone_type == 'ResNet50_SSL':
            patch_tokens = self.get_r50_ssl_features(inputs, target_backbone)
        elif self.backbone_type == 'EndoViT':
            patch_tokens = self.get_endovit_ssl_features(inputs, target_backbone)
        elif self.backbone_type == 'SurgeNet':
            patch_tokens = self.get_surgenet_ssl_features(inputs, self.backbone)
        else:
            features = target_backbone(inputs)
            patch_tokens, last_tokens = self.preprocess_features(features)

        # patch_tokens = patch_tokens.permute(0, 1, 2, 3, 4)
        latent = target_neck(patch_tokens)
        return latent.mean(dim=[2, 3])

    def loss(self, inputs: torch.Tensor, data_samples: SampleList,
             **kwargs) -> dict:        
        current_inputs, future_inputs = torch.split(inputs, 1, dim=1)
        current_inputs = current_inputs.view((-1, ) + current_inputs.shape[2:])
        future_inputs = future_inputs.view((-1, ) + future_inputs.shape[2:])
        
        if self.backbone_type == 'ResNet50_SSL':
            patch_tokens = self.get_r50_ssl_features(current_inputs, self.backbone)
        elif self.backbone_type == 'EndoViT':
            patch_tokens = self.get_endovit_ssl_features(current_inputs, self.backbone)
        elif self.backbone_type == 'SurgeNet':
            patch_tokens = self.get_surgenet_ssl_features(current_inputs, self.backbone)
        else:
            video_features = self.backbone(current_inputs)
            patch_tokens, last_tokens = self.preprocess_features(video_features)

        # patch_tokens = patch_tokens.permute(0, 2, 3, 4, 1)
        latent = self.neck(patch_tokens)
        latent = latent.mean(dim=[2, 3]) # spatial pooling

        with torch.no_grad():
            self.target_net.update_parameters(torch.nn.Sequential(self.backbone, self.neck))
            future_latent = self.process_future_inputs(future_inputs)

        # autoregressive prediction of future features
        future_pred_latent = self.imaginator(latent, future_steps=self.future_steps)
        future_mask = None

        if self.with_state_change_decoder:
            sc_dec_input = torch.cat([latent, future_pred_latent], dim=1)
            latent = self.state_change_decoder(sc_dec_input)

        losses = dict()
        # feat sim head
        loss_sc = self.cls_head.loss(latent, future_pred_latent, future_latent,
                                future_mask, self.future_steps, data_samples)
        if 'state_logits' in loss_sc:
            sc_logit = loss_sc.pop('state_logits')
        losses.update(loss_sc)
        
        return losses

    def predict(self, inputs: torch.Tensor, data_samples: SampleList,
                **kwargs) -> SampleList:
        current_inputs, future_inputs = torch.split(inputs, 1, dim=1)
        current_inputs = current_inputs.view((-1, ) + current_inputs.shape[2:])
        future_inputs = future_inputs.view((-1, ) + future_inputs.shape[2:])

        if self.backbone_type == 'ResNet50_SSL':
            patch_tokens = self.get_r50_ssl_features(current_inputs, self.backbone)
        elif self.backbone_type == 'EndoViT':
            patch_tokens = self.get_endovit_ssl_features(current_inputs, self.backbone)
        elif self.backbone_type == 'SurgeNet':
            patch_tokens = self.get_surgenet_ssl_features(current_inputs, self.backbone)
        else:
            video_features = self.backbone(current_inputs)
            patch_tokens, last_tokens = self.preprocess_features(video_features)

        # patch_tokens   = patch_tokens.permute(0, 2, 3, 4, 1)
        latent = self.neck(patch_tokens)
        latent = latent.mean(dim=[2, 3]) # spatial pooling

        with torch.no_grad():
            future_latent = self.process_future_inputs(future_inputs)

        future_pred_latent = self.imaginator(latent, future_steps=self.future_steps)        
        future_mask = None

        if self.with_state_change_decoder:
            sc_dec_input = torch.cat([latent, future_pred_latent], dim=1)
            latent = self.state_change_decoder(sc_dec_input)

        return self.cls_head.predict(latent, future_pred_latent, future_latent,
                                future_mask, self.future_steps, data_samples)
