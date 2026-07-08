_base_ = [
    '../../../_base_/models/vits_scadyn_256d_k25g.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

randomness = dict(seed=981)
load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'

model = dict(
    teacher_cfg='mmaction::state_change/centroid_transition/teacher/t_vits_videomae_verbsc_256d_k25g_seed981.py',
    teacher_ckpt='weights/t_vits_vm_vsc_k25g_1e4lr_ep27.pth',
    teacher_affinity='cosine_sim',
    sc_emb_loss=dict(type='l1_loss', loss_weight=1.0, enable=True),
    sc_logits_loss=dict(type='l1_loss', loss_weight=1.0, temp=0.3, anneal_temp=False, enable=True),
    centroid_transition=dict(
        _delete_=True,
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
    imaginator=dict(
        type='RNNFeaturePredictor',
        in_channels=384,
        hid_channels=1024,
        out_channels=384,
        model_type='GRU'),
    aux_head=dict(
        type='FutrReconHead',
        loss_feature=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=0.7))
    )

# train_dataloader = dict(
#     batch_size=16)

# val_dataloader = dict(
#     batch_size=16)

# test_dataloader=dict(
#     batch_size=16)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        indices=10,
    ))

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        indices=10,
    ))

# test_dataloader = dict(
#     batch_size=8,
#     dataset=dict(
#         indices=100,
#     ))