_base_ = [
    '../../../../_base_/models/vits_verbsc_256d.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

load_from = 'weights/vits_phase_8f_ep38_seed981.pth'
model = dict(use_futr=False)
randomness = dict(seed=981)

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