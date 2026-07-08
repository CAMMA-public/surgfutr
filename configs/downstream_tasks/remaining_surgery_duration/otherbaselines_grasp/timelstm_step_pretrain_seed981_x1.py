_base_ = [
    '../../../_base_/rarp_models/gsp_vits_phaserec_256d.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
]

randomness = dict(seed=981)

model = dict(
    type='BaselineActionAnticipatorSFPBench',
    backbone_ckpt='TIMELSTM',
    backbone_type='TIMELSTM',
    head_feature_src='backbone',
    freeze_backbone=False,
    cls_head=dict(
        _delete_=True,
        type='TemporalPhaseHead',
        input_dim=2048,
        num_classes=21,
        label_type='step',
        feat_agg='mean',
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

task = 'step-pretraining'
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        task=task))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        task=task))

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task))

test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        task=task))

val_evaluator = dict(_delete_=True, type='AccMetric')
test_evaluator = dict(_delete_=True, type='AccMetric')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='acc/top1',
        rule='greater',
        max_keep_ckpts=3))
