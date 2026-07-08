_base_ = ['../default_runtime.py',          
          '../datasets/mbp_anticipation_stras.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True
task = 'rsd'

model = dict(
    type='VideoClusterGraphWithFutr',
    freeze_backbone=False,
    use_futr=False,
    agg_feats='mean',
    backbone_type='VisionTransformer',
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
    clustering=None,
    graph=None,
    state_change_decoder=dict(
        type='StateChangeDecoder',
        input_dim=384, # 384
        mlp_scale=2,
        output_dim=128,
        clip_len=4, #clip_len//2,
        feature_size=(14, 14),
        fc_type='ft',
        num_mlp_layers=1,
        num_latent_actions=6,
        feature_to_use=(1, 0, 0), # (only cur, cur+futr, cur+futr+la)
        dropout=0.0,
        component='verb'),
    cls_head=dict(
        type='RSDPhaseTransitionHead',
        input_dim=384,
        output_dim=384,
        clip_len=4,
        num_classes=1,
        num_centroids=25,
        num_sc_classes=10,
        feat_type='backbone',
        feat_agg='flatten',
        loss_task=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', rule='less', max_keep_ckpts=2))
