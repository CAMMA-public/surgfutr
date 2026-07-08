_base_ = ['../default_runtime.py',          
          '../datasets/cholect50_phase.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True

model = dict(
    type='Recognizer3D',
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
        type='PhaseRecHead',
        num_classes=7,
        in_channels=384,
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

