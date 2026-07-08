dataset_type = 'CholecStateChangePretrainingDataset'
data_root = 'videos/ct50_state_change_clips'
ann_file_train = 'data/state_change/clip3s_verb_state_change_v1.json'
ann_file_val   = 'data/state_change/clip3s_verb_state_change_v1.json'
ann_file_test  = 'data/state_change/clip3s_verb_state_change_v1.json'
file_client_args = dict(io_backend='disk')

task = 'state-change'
extra_keys=('cl_v', 'cl_ivt', 'sc_v', 'sc_i', 'sc_ivt', 'sc_iv', 'sc_it', 'phase', 'futr_v', 'clip_meta', 'next_clip_filename')

train_pipeline = [
    dict(type='LoadCurFutrClip', **file_client_args),
    dict(type='ResizeCurFutr', scale=(-1, 256), input_key='', out_key=''),
    dict(
        type='StudentTransforms',
        transforms=[
            dict(type='PytorchVideoWrapperCurFutr', op='RandAugment', magnitude=7, num_layers=4, input_key='', out_key=''),
            dict(type='ResizeCurFutr', scale=(224, 224), input_key='', out_key='', keep_ratio=False),
    ]),
    dict(type='FormatTeacherStudentShape', input_format='NCTHW'),
    dict(type='PackTeacherStudentInputs', enable_ts=False, extra_keys=extra_keys)]

val_pipeline = [
    dict(type='LoadCurFutrClip', **file_client_args),
    dict(type='ResizeCurFutr', scale=(224, 224), input_key='', out_key='', keep_ratio=False), # check if keep_ratio=False is needed
    dict(
        type='StudentTransforms',
        transforms=[
            dict(type='ResizeCurFutr', scale=(224, 224), input_key='', out_key='', keep_ratio=False),
    ]),
    dict(type='FormatTeacherStudentShape', input_format='NCTHW'),
    dict(type='PackTeacherStudentInputs', extra_keys=extra_keys)]
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

val_evaluator = dict(
    type='FutureStateChangeMetric',
    metric_list=('multilabel_average_precision'),
    average='macro')
test_evaluator = val_evaluator