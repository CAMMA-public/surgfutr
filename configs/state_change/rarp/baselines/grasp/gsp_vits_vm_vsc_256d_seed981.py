_base_ = [
    '../../../../_base_/rarp_models/gsp_vits_verbsc_256d.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
model = dict(use_futr=False)
randomness = dict(seed=981)

train_dataloader = dict(
    batch_size=16)

val_dataloader = dict(
    batch_size=16)

test_dataloader = dict(
    batch_size=16)

# train_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=50,
#     ))

# val_dataloader = dict(
#     batch_size=4,
#     dataset=dict(
#         indices=10,
#     ))