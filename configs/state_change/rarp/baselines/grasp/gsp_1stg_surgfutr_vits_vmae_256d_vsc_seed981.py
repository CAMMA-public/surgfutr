_base_ = [
    '../../../../_base_/rarp_models/gsp_vits_surgfutr_256d.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

randomness = dict(seed=981)
load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
model = dict(
    base_momentum=0.004,
    future_steps=2,
    state_change_decoder=dict(
        type='StateChangeDecoder',
        input_dim=384, # 384
        mlp_scale=2,
        output_dim=128,
        clip_len=4,
        feature_size=(14, 14),
        fc_type='ft',
        num_classes=21,
        num_mlp_layers=1,
        num_latent_actions=6,
        feature_to_use=(1, 0, 0), # (only cur, cur+futr, cur+futr+la)
        dropout=0.0,
        component='verb'),
    imaginator=dict(
        hid_channels=1024),
    cls_head=dict(
        _delete_=True,
        type='SurgFutrStateChangeHead',
        feature_dim=384,
        input_dim=128,
        hidden_dim=512,
        output_dim=128,
        num_states=4,
        num_classes=10,
        num_centroids=25,
        seq_len=2, # no clustering and centroid transition
        component='verb',
        dataset='grasp',
        method=-1, # default linear classifier from sc embedding
        downsample_feature=False,
        pool_cur_futr=False,
        loss_state=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_feature=dict(_scope_='mmdet', type='SmoothL1Loss', loss_weight=1.0)))

train_dataloader = dict(
    batch_size=16)

val_dataloader = dict(
    batch_size=16)

test_dataloader = dict(
    batch_size=16)

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=50,
#     ))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#     ))

val_evaluator = dict(
    _delete_=True,
    type='SurgFutrStateChangeMetric',
    metric_list=('multilabel_average_precision'),
    average='macro')
test_evaluator = val_evaluator