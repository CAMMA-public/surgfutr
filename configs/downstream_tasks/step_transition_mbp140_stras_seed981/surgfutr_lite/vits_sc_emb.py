_base_ = [
    '../../../_base_/mbp_models/mbp_vits_anticipation_stras.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

# NOTE: should not be used as we are checking transfer from CholecT50 to MBP140, 
# but set num_sc_classes to 10 to match the number of classes in CholecT50

# load_from = 'weights/gsp_1stg_surgfutr_vits_256d_seed981_ep40.pth'
load_from = "weights/ts_vits_surgfutr_1024igtr_vsc_ep20.pth"

clip_len=8
model = dict(
    type='SCADynDownstreamRecognizer',
    future_steps=clip_len//2,
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
        num_classes=10,
        clip_len=clip_len),
    cls_head=dict(
        input_dim=128,
        feat_type='sc_cls_embs',
        feat_agg='flatten',
        clip_len=clip_len,
        num_sc_classes=10,
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
#         task=task,
#         clip_len=clip_len))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#         task=task,
#         clip_len=clip_len))

val_evaluator = dict(type='RSDPhaseTransitionMetric', 
            task=task,
            event_time_markers=[46, 82])
test_evaluator = val_evaluator
