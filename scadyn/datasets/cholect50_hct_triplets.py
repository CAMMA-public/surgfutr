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

@DATASETS.register_module()
class CT50_HCT_TripletsDataset(VideoDataset):
    """ Video Dataset for HCT Triplet Anticipation at window 1, 3, 5, 7"""
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
        data = load(osp.join(self.ann_file, f'clip3s_HCT_triplets_anticipation_v1.json'))[self.split]
        records = []
        label_list = []
        triplet_ids_selected = [58, 59, 57, 79, 78, 69, 68] #, 84, 34]
        # ['hook,dissect,cystic_duct', 'hook,dissect,cystic_plate', 'hook,dissect,cystic_artery', 
        # 'clipper,clip,cystic_duct', 'clipper,clip,cystic_artery', 'scissors,cut,cystic_duct', 'scissors,cut,cystic_artery']
        cls_idx_offset = 0
        bg_count=0

        null_label = [0] * len(triplet_ids_selected)
        if not self.disable_bg:
            null_label.append(0) # the first class is background
            cls_idx_offset = 1

        class_reindex = {t:i+cls_idx_offset for i, t in enumerate(triplet_ids_selected)} # adjust based on bg class
        
        for video_name, video_clip_list in data.items():
            for clip in video_clip_list:
                # check if clip['event_triplet_id'] has class 84 and 34
                if clip['event_triplet_id'] in [84, 34]: # NOTE: not present in validation set
                    # print(f'{video_name} has class 84 or 34')
                    continue

                if clip['anticipation_duration'] not in self.anticipation_window:
                    continue

                clip_label = deepcopy(null_label)
                clip['video_name'] = video_name
                try:
                    clip_label[class_reindex[clip['event_triplet_id']]] = 1
                except:
                    breakpoint()
                clip['label_onehot'] = clip_label
                clip['label'] = class_reindex[clip['event_triplet_id']]
                records.append(clip)
                label_list.append(class_reindex[clip['event_triplet_id']])

                if not self.disable_bg:
                    # create copies of the record until the window with label 0s # for train/val
                    for i in range(1, clip['anticipation_duration'], 1):

                        # test is only on windows 1, 3, 5
                        if self.split != 'train' and i in [2, 4, 6]:
                            continue

                        new_clip = deepcopy(clip)
                        new_clip['label_onehot'] = deepcopy(null_label)
                        new_clip['label_onehot'][0] = 1
                        new_clip['label'] = 0
                        new_clip['anticipation_duration'] = i
                        records.append(new_clip)
                        label_list.append(0)
                    
                    bg_count += 3

                    if bg_count > 70 and self.split == 'test':
                        self.disable_bg = True

                    if bg_count > 25 and self.split == 'val':
                        self.disable_bg = True

                    if bg_count > 180 and self.split == 'train':
                        self.disable_bg = True


        # create class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(label_list), y=label_list)
        # count how many records per class
        per_class_counts = np.bincount(label_list)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        print(f'Class weights: {class_weights} and total records: {len(records)}')
        print(f'Per class counts: {per_class_counts}')

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
            video_info['task'] = 'hct_triplets'
            video_info['video_name'] = rec['video_name']
            video_info['img_shape'] = self.img_shape
            video_info['verb_label'] = torch.tensor(rec['cur_verb'], dtype=torch.int)
            video_info['phase_label'] = torch.tensor(rec['cur_phase'], dtype=torch.int)
            video_info['triplet_label'] = torch.tensor(rec['cur_triplet'], dtype=torch.int)
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