_base_ = ['../default_runtime.py',
          '../datasets/cholect50_state_change_2s.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True

# load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
model = dict(
    type='BaselineActionAnticipator',
    freeze_backbone=False,
    use_futr=False,
    agg_feats='mean',
    backbone_type='VisionTransformer',
    model_name='AVT',
    backbone=dict(        
        type='CustomViT',
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        tubelet_size=2,
        num_frames=8,
        drop_path_rate=0.1, # check this or remove
        return_feat_map=False,
        return_feat_before_norm=True,
        norm_cfg=dict(type='LN', eps=1e-6)),
    neck=None,
    cls_head=dict(
        type='StateChangeClassHead2S',
        in_channels=128,
        out_channels=128,
        clip_len=4,
        num_states=2,
        component='verb',
        loss_state=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))



