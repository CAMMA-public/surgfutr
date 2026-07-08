_base_ = [
    '../../../_base_/models/vits_cholectrack_events_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_casea3.py']

randomness = dict(seed=981)
load_from = 'weights/avt_v2_nextact_seed981_x1_ep2.pth'

model = dict(
    type='BaselineActionAnticipatorSFPBench',
    backbone_ckpt='AVT',
    backbone_type='AVT',
    head_feature_src='decoder',
    backbone=dict(
        drop_path_rate=0.1
    ),
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        input_dim=2048,
        feat_type='imaginator_flatten1',
        feat_agg='none',
        num_classes=3,
        task='cholectrack-bleeding',
        clip_len=8)
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

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         disable_bg=True,
# ))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         disable_bg=True,
# ))


val_evaluator = dict(type='IAE_Metric', task=task, num_classes=3)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='iae/overall_mAP',
        rule='greater',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='iae/overall_mAP',
        rule='greater',
        patience=5,
        min_delta=1e-4,
        strict=False)
]