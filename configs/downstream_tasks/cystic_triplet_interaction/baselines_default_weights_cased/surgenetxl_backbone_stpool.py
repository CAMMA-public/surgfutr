_base_ = [
    '../../../_base_/models/vits_ct50_cystic_triplets_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_cased.py']

load_from = None

model = dict(
    type='CentroidTSDownstream',
    backbone_ckpt='-1',
    backbone_type='SurgeNet',
    use_centroid_transition=False,
    clustering=None,
    graph=None,
    state_change_decoder=None,
    aux_head=None,
    imaginator=None,
    cls_head=dict(
        _delete_=True,
        type='TimeConditionedAnticipationHead',
        input_dim=512,
        num_classes=7,
        num_centroids=25,
        feat_type='backbone',
        feat_agg='pool',
        clip_len=1)
)

task = 'hct-triplets'
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
        task=task,
        disable_bg=True))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=7)
test_evaluator = val_evaluator
