dataset_type = 'CT50_HCT_TripletsDataset'
data_root = 'videos/ct50_hct_triplet_clips'
anticipation_window = [1, 3, 5, 7]
task = 'hct-triplets'
ann_folder = 'data/hct_triplets'
file_client_args = dict(io_backend='disk')
extra_keys=('anticipation_window', 'task')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackTeacherStudentInputs', extra_keys=extra_keys)]  

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackTeacherStudentInputs', extra_keys=extra_keys)]
test_pipeline = val_pipeline

train_dataloader=dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_folder,
        clip_len=8,
        frame_interval=8,
        img_shape=(224, 224),
        data_prefix=dict(video=data_root),
        split='train',
        anticipation_window=anticipation_window,
        pipeline=train_pipeline))

val_dataloader=dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_folder,
        clip_len=8,
        frame_interval=8,
        data_prefix=dict(video=data_root),
        split='val',
        anticipation_window=anticipation_window,
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_folder,
        clip_len=8,
        frame_interval=8,
        data_prefix=dict(video=data_root),
        split='test',
        anticipation_window=anticipation_window,
        pipeline=test_pipeline,
        test_mode=True))
