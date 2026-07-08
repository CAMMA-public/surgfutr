dataset_type = 'CholecStateChangePretrainingDataset'
data_root = 'videos/ct50_state_change_clips'
ann_file_train = 'data/state_change/clip3s_verb_state_change_v1.json'
ann_file_val   = 'data/state_change/clip3s_verb_state_change_v1.json'
ann_file_test  = 'data/state_change/clip3s_verb_state_change_v1.json'
file_client_args = dict(io_backend='disk')
task='phase-pretraining'

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
    dict(type='PackActionInputs')]

# dict(type='PackTeacherStudentInputs', extra_keys=extra_keys)
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        task=task,
        clip_len=8,
        frame_interval=8,
        data_prefix=dict(video=data_root),
        split='train',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        task=task,
        clip_len=8,
        frame_interval=8,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        split='val',
        test_mode=True))

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        task=task,
        clip_len=8,
        frame_interval=8,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        split='test',
        test_mode=True))
