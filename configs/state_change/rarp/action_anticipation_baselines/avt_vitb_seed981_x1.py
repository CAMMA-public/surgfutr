_base_ = [
    '../../../_base_/rarp_models/gsp_baseline_action_anticipator.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
]

randomness = dict(seed=981)
model = dict(
    use_futr=True,
    backbone_ckpt='AVT',
    backbone_type='AVT',
    backbone=dict(
        drop_path_rate=0.1
    ),
    cls_head=dict(
        type='AVTHead',
        in_channels=128,
        out_channels=128,
        clip_len=4,
        num_states=2,
        component='verb',
        loss_state=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    )

train_dataloader=dict(
    batch_size=32,
    dataset=dict(
    ))

val_dataloader=dict(
    batch_size=32,
    dataset=dict(
    ))

test_dataloader=dict(
    batch_size=32,
    dataset=dict(
    ))

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

# val_dataloader=dict(
#     batch_size=2,
# )

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='mAP/state_mAP',
        rule='greater',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='mAP/state_mAP',
        rule='greater',
        patience=5,
        min_delta=1e-4,
        strict=False)
]