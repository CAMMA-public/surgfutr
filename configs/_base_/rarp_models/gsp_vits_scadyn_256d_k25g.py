_base_ = ['../default_runtime.py',          
          '../datasets/gsp_state_change.py',
          '../custom_imports.py',
]

randomness = dict(seed=981)
find_unused_parameters = True

model = dict(
    type='SCADyn',
    freeze_backbone=False,
    agg_feats='mean',
    teacher_cfg='mmaction::state_change/rarp/teacher/run001/ptd_t_vits_vm_vsc_256d_k25g_v0_seed981.py',
    teacher_ckpt='weights/ptd_t_vits_vm_vsc_k25g_v0_seed981_ep19.pth',
    sc_emb_loss=dict(type='l1_loss', loss_weight=1.0, enable=True),
    sc_logits_loss=dict(type='l1_loss', loss_weight=1.0, temp=0.1, anneal_temp=True, enable=True),
    g_feat_loss=dict(type='l1_loss', loss_weight=1.0, enable=False),
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
    centroid_transition=dict(
        type='ActionDynamicsModule',
        input_dim=384,
        hidden_dim=2048,
        output_dim=384,
        num_centroids=25,
        num_heads=4,
        num_layers=4,
        apply_heads=False,
        dropout=0.0, 
        alpha=1.0, # 0.1
        temperature=1.0,
        fn='transformer',
        centroid_update='default',
        target_transition='gumbel_softmax',
        loss_transition=dict(name='wasserstein_loss', p=2, blur=0.01, weight=1.0),
        loss_futr_pred=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=0.5)),
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
        num_classes=21,
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
        dataset='grasp',
        loss_state=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

train_dataloader = dict(
    batch_size=4)

