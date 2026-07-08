_base_ = [
    '../../../_base_/mbp_models/mbp_vits_anticipation_bern.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

# load_from = 'weights/gsp_1stg_surgfutr_vits_256d_seed981_ep40.pth'
load_from = "weights/ts_vits_surgfutr_1024igtr_vsc_ep20.pth"

clip_len = 8
model = dict(
    type='SCADynDownstreamRecognizer',
    future_steps=clip_len//2,
    backbone=dict(
        num_frames=clip_len),
    clustering=None,
    graph=None,
    state_change_decoder=None,
    imaginator=dict(
        type='RNNFeaturePredictor',
        in_channels=384,
        hid_channels=1024,
        out_channels=384,
        model_type='GRU'),
    cls_head=dict(
        input_dim=384,
        feat_type='imaginator_flatten', # NOTE: change
        feat_agg='flatten',
        num_classes=138,
        clip_len=clip_len//2)
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
