_base_ = ['../default_runtime.py',          
          '../datasets/cholectrack_event_anticipation.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True
task = 'cholectrack-bleeding'

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
    clustering=dict(
        type='SinkhornClustering',
        input_dim=384,
        output_dim=128,
        hidden_dim=2048,
        K=25,
        hard_assign=False,
        marginals_temp_r=1.0,
        marginals_temp_c=1.0,
        sinkhorn_iterations=3,
        sinkhorn_lambda=1.0,
        uniform_marginals=False),
    graph=dict(
        type='ClusterGraph',
        input_dim=384,
        hidden_dim=256,
        output_dim=256,
        bottleneck_dim=256,
        similarity_criteria='cdist_norm_exp', # 'cdist',
        similarity_threshold=0.02, # -0.5 for cdist, 0.9 for cosine (but they are mostly the same)
        weighted_adjacency=False,
        gcn_type='gat',
        num_heads=4,
        pool_heads=False,
        apply_linear_projection=True,
        num_graph_layers=1,
        use_readout=False,
        lp_choice='default',
        use_bias=False,
        activation='elu',
        temperature=0.05, # 0.1
        affine_fn_on_feats=False),
    state_change_decoder=dict(
        type='StateChangeDecoder',
        input_dim=256, # 384
        mlp_scale=2,
        output_dim=128,
        clip_len=25*4, #clip_len//2,
        feature_size=(14, 14),
        fc_type='ft',
        num_mlp_layers=1,
        num_latent_actions=6,
        feature_to_use=(1, 0, 0), # (only cur, cur+futr, cur+futr+la)
        dropout=0.0,
        component='verb'),
    cls_head=dict(
        type='TimeConditionedAnticipationHead',
        input_dim=384,
        output_dim=384,
        num_classes=2, # iae + background
        num_centroids=25,
        num_sc_classes=10,
        enable_loss_weight=True,
        normalize_windows=True,
        normal_clip_ratio=0.2,
        feat_type='backbone',
        feat_agg='flatten',
        loss_task=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', rule='greater', max_keep_ckpts=2))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=2)
test_evaluator = val_evaluator

