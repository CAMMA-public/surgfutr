_base_ = [
    '../../../_base_/models/baseline_action_anticipator.py',
    '../../../_base_/schedules/schedule_coslr_sgd_1x_highlr.py'
]

randomness = dict(seed=981)
# Replace with Stage-1 pretraining checkpoint path.
load_from = 'weights/ct50_sfrulstm_pretrain_seed981_ep16.pth'

model = dict(
    use_futr=True,
    backbone_ckpt='SLOWFAST_RULSTM',
    backbone_type='RULSTM',
    slowfast_ratio=2,
    backbone=dict(
        drop_path_rate=0.1
    ),
    cls_head=dict(
        _delete_=True,
        type='AVTV2Head',
        in_channels=1024,
        out_channels=1024,
        clip_len=4,
        num_classes=10,
        component='verb',
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    )

# train_dataloader=dict(
#     batch_size=16,
#     dataset=dict(
#     ))

# val_dataloader=dict(
#     batch_size=16,
#     dataset=dict(
#     ))

# train_dataloader=dict(
#     batch_size=2,
#     dataset=dict(
#         indices=50)
# )

# val_dataloader=dict(
#     batch_size=2,
#     dataset=dict(
#         indices=100)
# )


# test_dataloader=dict(
#     batch_size=16,
#     dataset=dict(
#     ))

train_dataloader=dict(
    batch_size=8,
    dataset=dict(
    ))

val_dataloader=dict(
    batch_size=1,
    dataset=dict(
    ))

test_dataloader=dict(
    batch_size=1,
    dataset=dict(
    ))

val_evaluator = dict(
    type='FeatClsMetric',
    metric_list=('multilabel_average_precision'),
    num_classes=10,
    eval_type=['cls'],
    average='macro')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='mAP/cls_mAP',
        rule='greater',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mAP/cls_mAP',
        rule='greater',
        patience=5,
        min_delta=1e-4,
        strict=False)
]
