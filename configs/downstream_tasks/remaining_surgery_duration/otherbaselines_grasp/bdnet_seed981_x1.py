_base_ = [
    '../../../_base_/rarp_models/gsp_vits_rsd_phase_transition.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

randomness = dict(seed=981)
model = dict(
    type='BaselineActionAnticipatorSFPBench',
    backbone_ckpt='BDNET',
    backbone_type='BDNET',
    head_feature_src='bdnet',
    bdnet_dncl_weight=0.001,
    bdnet_num_blstms=5,
    bdnet_dropouts=[0.0, 0.2, 0.4, 0.4, 0.6],
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        _delete_=True,
        type='RSDNetHead',
        input_dim=128,
        feat_agg='last',
        use_elapsed_time=False,
        use_progress_head=False,
        use_uncertainty_head=True,
        elapsed_scale=300.0,
        aux_label_key='phase',
        aux_num_classes=7,
        loss_weight_aux=0.5,
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