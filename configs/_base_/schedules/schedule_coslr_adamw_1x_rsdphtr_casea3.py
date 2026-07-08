train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=19, #40,
        eta_min=1e-7,
        by_epoch=True,
        begin=1,
        end=20,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2))

auto_scale_lr = dict(base_batch_size=64)