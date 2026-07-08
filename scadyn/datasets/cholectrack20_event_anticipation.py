from typing import Callable, List, Optional, Union
import os.path as osp
import numpy as np
from copy import deepcopy
import torch
from mmengine import load
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmaction.datasets import VideoDataset
from sklearn.utils.class_weight import compute_class_weight

# Rendezvous split
rdv_split = {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val'  : [8, 12, 29, 50, 78],
                'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            }
rdv_train_videos = [f'VID{i:02}' for i in rdv_split['train']]
rdv_val_videos = [f'VID{i:02}' for i in rdv_split['val']]
rdv_test_videos = [f'VID{i:02}' for i in rdv_split['test']]

@DATASETS.register_module()
class CholecTrack20_Event_Anticipation_Dataset(VideoDataset):
    """ Video Dataset for CholecTrack20 Event Anticipation at window 1, 3, 5"""
    def __init__(self,
                 ann_file: str='',
                 pipeline: List[Union[dict, Callable]]=[],
                 clip_duration: int = 3,
                 clip_len: int = 8,
                 task: str = 'hct-triplets',
                 frame_interval: int = 8,
                 sampling_fps: int = 25,
                 split: str = 'train',
                 disable_bg: bool = False,
                 anticipation_window: Optional[Union[int, List[int]]] = 1,
                 img_shape: List[int] = [224, 224],
                 data_prefix: ConfigType = dict(video=''),
                 start_index: int = 0,
                 test_mode: bool = False,
                 **kwargs) -> None:
    
        self.ann_file = ann_file
        self.task = task
        self.sampling_fps = sampling_fps
        self.frame_interval = frame_interval
        self.clip_len = clip_len
        self.clip_duration = clip_duration
        self.img_shape = img_shape
        self.total_frames = clip_duration * sampling_fps
        self.split = split
        if self.split == 'train':
            self.split = 'Training'
        elif self.split == 'val':
            self.split = 'Validation'
        elif self.split == 'test':
            self.split = 'Testing'
        
        self.anticipation_window = anticipation_window
        self.disable_bg = disable_bg

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            start_index=start_index,
            **kwargs)
        
    def load_annotations(self) -> List[dict]:
        ann_file = osp.join(self.ann_file, f'cholectrack20_events_antr_v2.json')
        data = load(ann_file)[self.split]

        records = []
        label_list = []
        null_label = [0] * 3 # first index is bleeding, second is smoke, third is occlusion

        for clip_name, clip in data.items():
            clip_label = deepcopy(null_label)
            label = clip['label']
            clip['video_name'] = clip_name
            clip['label_onehot'] = clip_label
            
            if label == 'bleeding':
                clip['label_onehot'][0] = 1
                clip['label'] = 0
            elif label == 'smoke':
                clip['label_onehot'][1] = 1
                clip['label'] = 1
            elif label == 'occluded':
                clip['label_onehot'][2] = 1
                clip['label'] = 2

            records.append(clip)
            label_list.append(clip['label'])

        # create class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(label_list), y=label_list)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        print(f'Class weights: {class_weights} and total records: {len(records)}')

        return records

    def load_data_list(self) -> List[dict]:
        """Load data list from the annotation file."""
        data = self.load_annotations()
        anticipation_window_list = []

        data_list = []
        for rec in data:
            video_info = dict()
            video_info['filename'] = osp.join(self.data_prefix['video'], rec['clip_id'])
            video_info['label'] = torch.tensor(rec['label'], dtype=torch.int)
            video_info['anticipation_window'] = torch.tensor(rec['anticipation_duration'], dtype=torch.int).unsqueeze(0)
            anticipation_window_list.append(rec['anticipation_duration'])
            video_info['task'] = 'cholectrack-bleeding'
            video_info['video_name'] = rec['video_name']
            video_info['img_shape'] = self.img_shape
            data_list.append(video_info)

        # check anticipation class again
        anticipation_classes = []
        for d in data_list:
            anticipation_classes.append(d['label'])
        print(f'Anticipation classes: {np.unique(anticipation_classes)}')
        print(f'Anticipation windows: {np.unique(anticipation_window_list)}')

        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        return data_info