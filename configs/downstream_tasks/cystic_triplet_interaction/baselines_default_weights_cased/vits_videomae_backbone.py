_base_ = [
    '../../../_base_/models/vits_ct50_cystic_triplets_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_cased.py']

load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'

model = dict(
    type='CentroidTSDownstream',
    use_centroid_transition=False,
    clustering=None,
    centroid_transition=None,
    graph=None,
    state_change_decoder=None,
    imaginator=None,
    aux_head=None,
    cls_head=dict(
        _delete_=True,
        type='TimeConditionedAnticipationHead',
        input_dim=384,
        clip_len=4,
        num_classes=7,
        num_centroids=25,
        feat_type='backbone',
        feat_agg='flatten'))

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
#         task=task,
#         indices=100))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         task=task))

test_dataloader = dict(
    dataset=dict(
        task=task,
        disable_bg=True))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=7)
test_evaluator = val_evaluator