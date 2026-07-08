_base_ = [
    '../../../../_base_/rarp_models/gsp_surgenetxl_verbsc_256d.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]


load_from=None
randomness = dict(seed=981)
model = dict(
    use_futr=False,
    cls_head=dict(
        clip_len=4,
        dataset='grasp'),
    data_preprocessor=dict(
        _delete_=True,
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW')
)

# train_dataloader = dict(
#     batch_size=16)

# val_dataloader = dict(
#     batch_size=16)

# test_dataloader = dict(
#     batch_size=16)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        indices=50,
    ))

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        indices=10,
    ))