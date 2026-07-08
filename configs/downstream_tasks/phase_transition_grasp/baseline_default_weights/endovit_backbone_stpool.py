_base_ = [
    '../../../_base_/rarp_models/gsp_vits_rsd_phase_transition.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = None

model = dict(
    backbone_ckpt='-1',
    backbone_type='EndoViT',
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
        feat_agg='pool',
        input_dim=768,
        clip_len=1,
        num_classes=33)
)

task = 'phase-transition'
clip_len = 4
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        task=task,
        clip_len=clip_len))

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         clip_len=clip_len,
#         task=task))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         clip_len=clip_len,
#         task=task))

# test_dataloader = dict(
#     dataset=dict(
#         task=task))

val_evaluator = dict(type='RSDPhaseTransitionMetric', 
                     task=task,
                     event_time_markers=[11, 22])
test_evaluator = val_evaluator
