_base_ = [
    '../../../_base_/models/vits_cholectrack_events_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_casea3.py']

load_from = 'weights/ts_vits_k25g_igtr_m2p5_s18_ep16.pth'

model = dict(
    type='SCADynDownstreamRecognizer',
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
        loss_futr_pred=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=0.5)),
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
        num_classes=3,
        task='cholectrack-bleeding')
)

task = 'cholectrack-bleeding'
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        disable_bg=True))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        task=task,
        disable_bg=True))

test_dataloader = dict(
    dataset=dict(
        task=task,
        disable_bg=True))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=3)
test_evaluator = val_evaluator
