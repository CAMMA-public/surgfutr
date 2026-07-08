_base_ = [
    '../../../_base_/models/vits_rsd_phase_transition_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

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
        num_classes=21,
        clip_len=8)
)

task = 'phase-transition'
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task))

val_dataloader = dict(
    batch_size=32,
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
    dataset=dict(
        task=task))

val_evaluator = dict(type='RSDPhaseTransitionMetric', task=task)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='long_term_anticipation/mae_phase-transition',
        rule='less',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='long_term_anticipation/mae_phase-transition',
        rule='less',
        patience=5,
        min_delta=1e-4,
        strict=False)
]