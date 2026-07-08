_base_ = [
    '../../../_base_/models/vits_rsd_phase_transition_256d.py',
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
        feat_agg='flatten',
        input_dim=768,
        clip_len=8,
        num_classes=1)
)

task = 'rsd'
# train_dataloader = dict(
#     batch_size=32,
#     dataset=dict(
#         task=task))

# val_dataloader = dict(
#     batch_size=32,
#     dataset=dict(
#         task=task))

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

# test_dataloader = dict(
#     dataset=dict(
#         task=task))

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

val_evaluator = dict(type='RSDPhaseTransitionMetric', task=task)
test_evaluator = val_evaluator
