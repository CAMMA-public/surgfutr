_base_ = [
    '../../../_base_/models/vits_cholectrack_events_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_casea3.py']

load_from = 'weights/ts_vits_k25g_igtr_m2p5_s18_ep16.pth'

model = dict(
    type='SCADynDownstreamRecognizer',
    clustering=None,
    graph=None,
    state_change_decoder=None,
    cls_head=dict(
        feat_type='backbone',
        feat_agg='flatten',
        num_classes=3,
        task='cholectrack-bleeding')
)

task = 'cholectrack-bleeding'
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

test_dataloader = dict(
    dataset=dict(
        task=task,
        disable_bg=True))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=3)
test_evaluator = val_evaluator
