_base_ = [
    '../../../_base_/models/vits_verbsc_256d_k25g.py',
    '../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
]

randomness = dict(seed=981)
load_from = 'weights/vit-small-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-25c748fd.pth'
model = dict(
    use_futr=True,
    backbone=dict(
        drop_path_rate=0.1
    ),
    clustering=dict(K=25),
    state_change_decoder=dict(
        input_dim=256,
        clip_len=25*4)
    )

# train_dataloader=dict(
#     batch_size=32)

# val_dataloader=dict(
#     batch_size=32)

train_dataloader=dict(
    batch_size=2,
    dataset=dict(
        indices=50
    )
)

val_dataloader=dict(
    batch_size=2,
    dataset=dict(
        indices=50
    )
)