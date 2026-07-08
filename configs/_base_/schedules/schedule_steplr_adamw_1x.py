train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        begin=0,
        end=5),
    dict(
        type='MultiStepLR',
        milestones=[12, 20],
        gamma=0.1,
        by_epoch=True,
        begin=5,
        end=30)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05))

auto_scale_lr = dict(enable=False, base_batch_size=64)