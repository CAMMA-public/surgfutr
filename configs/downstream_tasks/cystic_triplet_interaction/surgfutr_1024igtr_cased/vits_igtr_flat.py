_base_ = [
    '../../../_base_/models/vits_ct50_hct_triplets_256d.py',
     '../../../_base_/schedules/schedule_coslr_adamw_1x_rsdphtr_cased.py']

load_from = 'weights/ts_vits_surgfutr_1024igtr_vsc_ep20.pth'

model = dict(
    type='SCADynDownstreamRecognizer',
    clustering=None,
    graph=None,
    imaginator=dict(
        type='RNNFeaturePredictor',
        in_channels=384,
        hid_channels=1024,
        out_channels=384,
        model_type='GRU'),
    state_change_decoder=None,
    cls_head=dict(
        input_dim=384,
        feat_type='imaginator_flatten',
        feat_agg='flatten',
        num_classes=7)
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

test_dataloader = dict(
    dataset=dict(
        task=task,
        disable_bg=True))

val_evaluator = dict(type='IAE_Metric', task=task, num_classes=7)
test_evaluator = val_evaluator
