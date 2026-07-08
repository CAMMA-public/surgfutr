_base_ = ['../default_runtime.py',          
          '../datasets/gsp_state_change.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True

model = dict(
    type='SurgFutrPretrainer',
    freeze_backbone=False,
    future_steps=2,
    base_momentum=0.004,
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
        num_frames=4,
        drop_path_rate=0.1, # check this or remove
        return_feat_map=False,
        return_feat_before_norm=True,
        norm_cfg=dict(type='LN', eps=1e-6)),
    neck=None,
    imaginator=dict(
        type='RNNFeaturePredictor',
        in_channels=384,
        hid_channels=512,
        out_channels=384,
        model_type='GRU'),
    cls_head=dict(
        type='SurgFutrHead',
        in_channels=384,
        out_channels=384,
        clip_len=4,
        component='verb',
        loss_feature=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

val_evaluator = dict(
    type='FeaturePredictionAndClassificationMetric',
    metric_list=('multilabel_average_precision'),
    num_classes=10,
    average='macro')
test_evaluator = val_evaluator



