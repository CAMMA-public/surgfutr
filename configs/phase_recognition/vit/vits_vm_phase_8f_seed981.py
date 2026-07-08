_base_ = [
    '../../_base_/models/vits_phaserec_256d.py',
    '../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

randomness = dict(seed=981)
load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_dataloader = dict(
    batch_size=16)

val_dataloader = dict(
    batch_size=16)

# train_dataloader = dict(
#     batch_size=2,
#     dataset=dict(
#         indices=10))

# val_dataloader = dict(
#     batch_size=2,
#     dataset=dict(
#         indices=10))

test_dataloader=dict(
    batch_size=16)