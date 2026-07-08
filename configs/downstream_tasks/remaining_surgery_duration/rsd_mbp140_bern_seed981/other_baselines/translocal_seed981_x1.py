_base_ = [
    '../../../../_base_/mbp_models/mbp_vits_anticipation_bern.py',
     '../../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

randomness = dict(seed=981)

# Replace with your phase-pretrained TransLocal checkpoint path.
load_from = 'weights/ct50_translocal_phasept_seed981_ep32.pth'

model = dict(
    type='BaselineActionAnticipatorSFPBench',
    backbone_ckpt='TRANSLOCAL',
    backbone_type='TRANSLOCAL',
    head_feature_src='backbone',
    freeze_backbone=False,
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        _delete_=True,
        type='RSDNetHead',
        input_dim=2048,
        feat_agg='last',
        use_elapsed_time=True,
        elapsed_scale=600.0,
        loss_weight_rsd=1.0,
        loss_weight_progress=1.0)
)

task = 'rsd'
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
        save_best='long_term_anticipation/mae_rsd',
        rule='less',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='long_term_anticipation/mae_rsd',
        rule='less',
        patience=5,
        min_delta=1e-4,
        strict=False)
]
