_base_ = [
    '../../../../_base_/rarp_models/gsp_vits_verbsc_256d.py',
    '../../../../_base_/schedules/schedule_coslr_adamw_1x_highlr.py'
    ]

# TODO: check with stage 3 features
randomness = dict(seed=981)
model = dict(
    backbone_ckpt='weights/moco_cholec80_r50_pretrained.torch',
    backbone_type='ResNet50_SSL',
    backbone=dict(
        _delete_=True,
        type='ResNet50_SSL',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    state_change_decoder=dict(
        input_dim=2048,
        feature_size=(7, 7),
        num_classes=21,
        clip_len=4),
    cls_head=dict(
        clip_len=4,
        dataset='grasp'),
    data_preprocessor=dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    )

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