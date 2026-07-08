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
        type='SGD', lr=5e-2, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=64)