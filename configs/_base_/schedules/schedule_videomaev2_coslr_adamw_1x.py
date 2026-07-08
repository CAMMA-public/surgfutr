train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=8,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=32, # 40 - 8 warmup
        eta_min=1e-6,
        by_epoch=True,
        begin=8,
        end=40,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05),
    # Module-wise LR control (effective_lr = base_lr * lr_mult).
    # Tune these multipliers per experiment without changing model code.
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys=dict(
            backbone=dict(lr_mult=0.25),          # effective 5e-5 for pretrained ViT-S
            pos_embed=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            register_tokens=dict(decay_mult=0.0),
            img_mod_embed=dict(decay_mult=0.0),
            video_mod_embed=dict(decay_mult=0.0),
            neck=dict(lr_mult=1.0),                # 1e-4
            sinkhorn_clustering=dict(lr_mult=1.0), # 1e-4
            cluster_graph=dict(lr_mult=1.0),       # 1e-4
            state_change_decoder=dict(lr_mult=1.0),# 1e-4
            cls_head=dict(lr_mult=1.0),            # 1e-4
            imaginator=dict(lr_mult=1.0),          # 1e-4
            centroid_transition=dict(lr_mult=1.0), # 1e-4
        )),
    clip_grad=dict(max_norm=1.0, norm_type=2))

auto_scale_lr = dict(base_batch_size=64)
