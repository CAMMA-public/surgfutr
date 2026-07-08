_base_ = [
    '../../../../_base_/models/surgenetxl_verbsc.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

load_from=None
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