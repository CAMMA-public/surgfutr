_base_ = [
    '../../../_base_/rarp_models/gsp_vits_rsd_phase_transition.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = 'weights/gsp_1stg_surgfutr_vits_256d_seed981_ep40.pth'

clip_len=4
model = dict(
    type='SCADynDownstreamRecognizer',
    future_steps=2,
    backbone=dict(
        num_frames=clip_len),
    agg_feats='flatten',
    clustering=None,
    centroid_transition=None,
    graph=None,
    imaginator=dict(
        type='RNNFeaturePredictor',
        in_channels=384,
        hid_channels=1024,
        out_channels=384,
        model_type='GRU'),
    state_change_decoder=dict(
        input_dim=384,
        num_mlp_layers=1,
        num_classes=21,
        clip_len=clip_len),
    cls_head=dict(
        input_dim=128,
        feat_type='sc_cls_embs',
        feat_agg='flatten',
        clip_len=clip_len,
        num_sc_classes=21,
        num_classes=1)
)

task = 'rsd'
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
#         task=task,
#         clip_len=clip_len))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         clip_len=clip_len))

val_evaluator = dict(type='RSDPhaseTransitionMetric', task=task)
test_evaluator = val_evaluator
