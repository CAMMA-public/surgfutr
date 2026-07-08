from typing import Callable, List, Optional, Union
import os.path as osp
from copy import deepcopy
import collections
import numpy as np
import torch
from mmengine.fileio import exists
from mmengine import load
from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from mmaction.datasets import VideoDataset
from .utils import (prosta_to_components, gen_state_change_labels, StateTracker, reverse_map)
from sklearn.utils.class_weight import compute_class_weight
MAX_TRACKS=5
bile_duct_triplet_ids = [68, 69, 70, 78, 79, 80, 81]

class MBP140AnticipationClipRecord(object):
    def __init__(self, clipdata=None, dur=3):
        self.data = {}
        self.clipname = clipdata['clip_id']
        self.video, start_meta, end_meta = self.clipname.split('_')
        self.start_frame = int(start_meta[2:])
        self.end_frame   = int(end_meta[2:])

        self.update("rsd", clipdata['rsd'])
        self.update("phase_transition", clipdata['phase_transition'])
        self.update("step_transition", clipdata['step_transition'])

        # next clip
        self.update("next_clip_name", clipdata['futr_clip_name'])

    def update(self, key, value):
        self.data[key] = value

    @property
    def clipinfo(self):
        return self.clipname
    
    @property
    def next_clip_name(self):
        return self.data["next_clip_name"] # name of the next clip
    
    @property
    def rsd(self):
        return self.data["rsd"] # in minutes

    @property
    def phtr(self):
        return self.data["phase_transition"] # in seconds
    
    @property
    def steptr(self):
        return self.data["step_transition"] # in seconds

@DATASETS.register_module()
class MBP140AnticipationDataset(VideoDataset):
    """
    Video dataset for downstream tasks on GraSP dataset
    - rsd (remaining surgery duration)
    - phase-transition (phase transition)
    - verb-anticipation (verb component of the triplet)
    - state-change (verb component of the triplet)
    """
    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 task: str = 'state-change',
                 num_states: int = 4,
                 clip_dur: int = 4,
                 clip_len: int = 16,
                 frame_interval: int = 5,
                 frame_idx_offset: int = 10,
                 enable_fraction_conti: bool = False,
                 sampling_fps: int = 25,
                 frame_sampling_mode: str = 'uniform',
                 split: str = 'train',
                 num_train_videos: int = -1,
                 img_shape: List[int] = [448, 256],
                 data_prefix: ConfigType = dict(video=''),
                 start_index: int = 0,
                 test_mode: bool = False,
                 **kwargs) -> None:
        
        self.split = split
        self.clip_dur = clip_dur
        self.task = task
        self.sampling_fps = sampling_fps
        self.frame_idx_offset = frame_idx_offset
        self.frame_sampling_mode = frame_sampling_mode
        self.num_states = num_states
        self.frame_interval = frame_interval
        self.num_train_videos = num_train_videos
        self.clip_len = clip_len
        self.img_shape = img_shape
        self.enable_fraction_conti = enable_fraction_conti
        self.num_triplet_classes = len(reverse_map['ivt'])

        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            start_index=start_index,
            **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        allclipdata = load(self.ann_file)['anticipation_labels'][self.split]
        records, total_clips = [], 0
        records_by_vid = collections.OrderedDict()
        for vid in sorted(list(allclipdata.keys())): # videos in the split
            for clipdata in allclipdata[vid]:
                clip_record = MBP140AnticipationClipRecord(clipdata, dur=self.clip_dur)
                records.append(clip_record)
                records_by_vid.setdefault(clip_record.video, []).append(clip_record)
                total_clips += 1
        print(f"Processing # videos {len(records_by_vid)} with # clips {total_clips}...")
        
        records = []
        total_clips, missing_videos = 0, 0
        for vid_index, (vid, vrecords) in enumerate(records_by_vid.items()):
            vrecords = sorted(vrecords, key=lambda record: record.end_frame)
            cnt = 0
            for rec_idx, rec in enumerate(vrecords):
                nextclipfile = osp.join(self.data_prefix['video'], rec.next_clip_name)

                # check if the next clip file exists. Since this is frame based, iterate through frames and check if the frame exists
                futr_clip_meta = rec.next_clip_name.split('_')
                futr_clip_start = int(futr_clip_meta[1].split('ps')[-1])
                futr_clip_end = int(futr_clip_meta[2].split('pe')[-1])
                video_name = futr_clip_meta[0]
                video_dir = osp.join(self.data_prefix['video'], "frames", video_name)
                for frame_idx in range(futr_clip_start, futr_clip_end+1):
                    frame_file = osp.join(video_dir, f"{video_name}_{frame_idx:08d}.jpg") # 5550090_00000679.jpg or BP01-10156_00002205.jpg
                    if not exists(frame_file):
                        missing_videos += 1
                        print(f"Missing video clip: {nextclipfile}")
                        continue

                clip_record = deepcopy(rec)
                clip_record.rsd_label = rec.rsd
                clip_record.phtr_label = rec.phtr # 33
                clip_record.steptr_label = rec.steptr # 63
                clip_record.futr_clip_meta = [futr_clip_start, futr_clip_end]
                clip_record.anticipation_window = [3]
                clip_record.video_id = vid
                clip_record.clip_meta = [rec.start_frame, rec.end_frame]
                clip_record.next_clip_file = nextclipfile
                # if cnt > 4:
                #     breakpoint()
                """
                The gt pickle file from MBP140 provides for each video a custom start frame and custom end frame and that can cause the rsd to change 
                refering to this file that used to create the rsd and anticiption labels: preprocessing/state_change_analysis/mbp140/get_anticipation_labels.py
                We will use the fixed scaling of 600 for rsdnet and timelstm models as they need elapsed time. 
                """
                records.append(clip_record)
                cnt += 1

            if self.num_train_videos > 0 and vid_index >= self.num_train_videos:
                break

        print(f"# missing videos: {missing_videos}")
        print(f"Using # videos {len(records)} for {self.split} split...")
        return records

    def parse_records(self, vrec):
        # 1fps frame selection, no clip involved in prosta
        current_frame_ids = np.arange(vrec.clip_meta[0], vrec.clip_meta[1]+1)
        future_frame_ids = np.arange(vrec.futr_clip_meta[0], vrec.futr_clip_meta[1]+1)
        return current_frame_ids, future_frame_ids
        
    def load_data_list(self) -> List[dict]:
        exists(self.ann_file)
        self.data = self.load_annotations()
        data_list = []
        mismatched_frames = 0
        for vrecord in self.data:
            current_frame_ids, future_frame_ids = self.parse_records(vrecord)

            video_info = dict()
            # metas
            video_info['video_id']  = vrecord.video_id
            video_info['clip_meta'] = torch.tensor(vrecord.clip_meta, dtype=torch.float32).unsqueeze(0)
            video_info['anticipation_window'] = torch.tensor(vrecord.anticipation_window)
            video_info['filename']  = osp.join(self.data_prefix['video'], vrecord.clipname)
            video_info['next_clip_filename'] = vrecord.next_clip_file
            video_info['img_shape'] = self.img_shape
            video_info['task'] = self.task
            video_info['data_prefix'] = self.data_prefix['video']
            video_info['frame_inds']   = torch.tensor([int(p) for p in current_frame_ids], dtype=torch.int32)
            video_info['future_frame_inds'] = torch.tensor([int(p) for p in future_frame_ids], dtype=torch.int32)
            
            if self.task == 'rsd':
                video_info['label']  = torch.tensor(vrecord.rsd_label).unsqueeze(0)
            elif self.task == 'phase-transition':
                video_info['label']  = torch.tensor(vrecord.phtr_label).unsqueeze(0)
            elif self.task == 'step-transition':
                video_info['label']  = torch.tensor(vrecord.steptr_label).unsqueeze(0)

            data_list.append(video_info)
        print(f"Total number of samples in dataset: {len(data_list)}")
        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        return data_info