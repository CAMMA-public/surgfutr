_base_ = [
    '../../../_base_/models/vits_verbsc_256d_k25g.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]
load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
randomness = dict(seed=981)
model = dict(use_futr=False)

# train_dataloader = dict(
#     batch_size=16)

# val_dataloader = dict(
#     batch_size=16)

# test_dataloader=dict(
#     batch_size=16)

train_dataloader=dict(
    batch_size=8,
    dataset=dict(
    ))

val_dataloader=dict(
    batch_size=1,
    dataset=dict(
    ))

test_dataloader=dict(
    batch_size=1,
    dataset=dict(
    ))