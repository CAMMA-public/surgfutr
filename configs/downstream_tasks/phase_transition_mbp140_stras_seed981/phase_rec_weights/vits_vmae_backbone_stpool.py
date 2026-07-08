_base_ = [
    '../../../_base_/mbp_models/mbp_vits_anticipation_stras.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = "weights/vits_phase_8f_ep38_seed981.pth"
randomness = dict(seed=981)

model = dict(
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
        feat_agg='pool',
        num_classes=36,
        clip_len=1)
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
    batch_size=32,
    dataset=dict(
        task=task))

val_evaluator = dict(type='RSDPhaseTransitionMetric', 
            task=task,
            event_time_markers=[12, 24])
test_evaluator = val_evaluator
