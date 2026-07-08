_base_ = [
    '../../../_base_/models/vits_rsd_phase_transition_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = None

model = dict(
    backbone_ckpt='weights/moco_cholec80_r50_pretrained.torch',
    backbone_type='ResNet50_SSL',
    backbone=dict(
        _delete_=True,
        type='ResNet50_SSL',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
        feat_agg='pool',
        input_dim=2048,
        clip_len=1,
        num_classes=21)
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
