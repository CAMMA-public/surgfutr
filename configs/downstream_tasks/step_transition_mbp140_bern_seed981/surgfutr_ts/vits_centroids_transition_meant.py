_base_ = [
    '../../../_base_/mbp_models/mbp_vits_anticipation_bern.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

# load_from = 'weights/gsp_ts_vits_k25g_v6_seed981_ep30.pth'
load_from = 'weights/ts_vits_k25g_igtr_m2p5_s18_ep16.pth'

clip_len = 8
model = dict(
    type='SCADynDownstreamRecognizer',
    backbone=dict(
        num_frames=clip_len),
    clustering=dict(
        _delete_=True,
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
        type='ClusteringTransitionPredictor',
        input_dim=384,
        hidden_dim=512,
        output_dim=384,
        num_centroids=25,
        num_heads=4,
        num_layers=4,
        apply_heads=False,
        dropout=0.0, 
        alpha=1.0,
        temperature=1.0,
        anneal_temp=False,
        fn='graph2',
        centroid_update='default',
        target_transition='default',
        transition_type='affinity',
        patch_topk=5, #784,
        loss_transition=dict(name='wasserstein_loss', p=2, blur=0.01, weight=1.0),
        loss_futr_pred=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=0.2)),
    graph=dict(
        _delete_=True,
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
    state_change_decoder=None,
    cls_head=dict(
        input_dim=256,
        feat_type='centroids_mean_temporal',
        feat_agg='flatten',
        num_classes=138)
)

task = 'step-transition'
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         clip_len=clip_len))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         clip_len=clip_len))

# test_dataloader = dict(
#     batch_size=32,
#     dataset=dict(
#         task=task,
#         clip_len=clip_len))

val_evaluator = dict(type='RSDPhaseTransitionMetric', 
            task=task,
            event_time_markers=[46, 82])
test_evaluator = val_evaluator
