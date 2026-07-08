_base_ = [
    '../../../_base_/mbp_models/mbp_vits_anticipation_bern.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
clip_len = 8
model = dict(
    backbone=dict(
        num_frames=clip_len),
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
        feat_agg='pool',
        clip_len=1,
        num_classes=138)
)

task = 'step-transition'
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
                     event_time_markers=[46, 82])
test_evaluator = val_evaluator
