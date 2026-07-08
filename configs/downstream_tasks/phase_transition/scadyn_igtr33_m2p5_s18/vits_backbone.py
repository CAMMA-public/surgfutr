_base_ = [
    '../../../_base_/models/vits_rsd_phase_transition_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr.py']

load_from = 'weights/ts_vits_k25g_igtr_m2p5_s18_ep16.pth'

model = dict(
    type='SCADynDownstreamRecognizer',
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
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

test_dataloader = dict(
    dataset=dict(
        task=task))

val_evaluator = dict(type='RSDPhaseTransitionMetric', task=task)
test_evaluator = val_evaluator
