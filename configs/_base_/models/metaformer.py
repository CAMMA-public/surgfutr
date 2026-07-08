_base_ = ['../default_runtime.py',          
          '../datasets/cholect50_state_change.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True

model = dict(
    type='VideoClusterGraphWithFutr',
    freeze_backbone=False,
    use_futr=False,
    agg_feats='mean',
    backbone_ckpt='-1',
    backbone_type='MetaFormer',
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
        input_dim=384,
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
        type='StateChangeClassHead',
        in_channels=128,
        out_channels=128,
        clip_len=4,
        num_states=4,
        component='verb',
        loss_state=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))