_base_ = [
    '../../../_base_/models/vits_rsd_phase_transition_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = 'weights/ts_vits_k25g_igtr_m2p5_s18_ep16.pth'

model = dict(
    type='SCADynDownstreamRecognizer',
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
