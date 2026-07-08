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

class GSPClipRecord(object):
    def __init__(self, clipdata=None, dur=3):
        self.data = {}
        self.clipname = clipdata['clip_id']
        self.video, start_meta, end_meta = self.clipname.split('_')
        self.start_frame = int(start_meta[2:])
        self.end_frame   = int(end_meta[2:])

        # check if the duration is 3 seconds
        assert self.end_frame - self.start_frame == dur, f"Duration is not {dur} seconds"
        assert clipdata['futr_end_t'] - clipdata['futr_start_t'] == dur, f"Future duration is not {dur} seconds"

        self.update("cur_v", clipdata['cur_verb'])
        self.update("cur_ivt", clipdata['cur_triplet'])
        if "cur_phase" in clipdata:
            self.update("cur_p", clipdata['cur_phase'])
        if "futr_phase" in clipdata:
            self.update("futr_p", clipdata['futr_phase'])
        self.update("futr_v", clipdata['futr_verb'])
        self.update("futr_ivt", clipdata['futr_triplet'])

        # next clip
        self.update("next_clip_name", clipdata['futr_clip_name'])

        # TODO:
        if "rsd" in clipdata:
            self.update("remaining_surgery_duration", clipdata["rsd"])

        if "phase_transition" in clipdata:
            self.update("phase_transition", clipdata["phase_transition"])

    def update(self, key, value):
        self.data[key] = value

    @property
    def clipinfo(self):
        return self.clipname
    
    @property
    def cur_phase(self):
        return self.data['cur_p']
    
    @property
    def cur_verb(self):
        return self.data['cur_v']
    
    @property
    def cur_triplet(self):
        return self.data['cur_ivt']
    
    @property
    def futr_verb(self):
        return self.data['futr_v'] # future_triplet
    
    @property
    def futr_triplet(self):
        return self.data['futr_ivt']
    
    @property
    def futr_phase(self):
        return self.data['futr_p']
    
    @property
    def next_clip_name(self):
        return self.data["next_clip_name"] # name of the next clip
    
    @property
    def rsd(self):
        if "remaining_surgery_duration" in self.data:
            return self.data["remaining_surgery_duration"] # in seconds
        else:
            return None

    @property
    def phtr(self):
        if "phase_transition" in self.data:
            return self.data["phase_transition"] # in seconds
        else:
            return None

@DATASETS.register_module()
class GraSPStepsStateChangePretrainingDataset(VideoDataset):
    """Video dataset for state change prediction on verb component of the triplet"""
    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 task: str = 'state-change',
                 num_states: int = 4,
                 clip_dur: int = 3,
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
        allclipdata = load(self.ann_file)[self.split]
        records, total_clips = [], 0
        records_by_vid = collections.OrderedDict()
        for vid in sorted(list(allclipdata.keys())): # videos in the split
            for clipdata in allclipdata[vid]:
                clip_record = GSPClipRecord(clipdata, dur=self.clip_dur)
                records.append(clip_record)
                records_by_vid.setdefault(clip_record.video, []).append(clip_record)
                total_clips += 1
        print(f"Processing # videos {len(records_by_vid)} with # clips {total_clips}...")
        
        records = []
        state_tracker = StateTracker()
        unique_phases = set()
        total_clips, missing_videos, sc_labels_list_v, sc_labels_list_ivt = 0, 0, [], []
        for vid_index, (vid, vrecords) in enumerate(records_by_vid.items()):
            vrecords = sorted(vrecords, key=lambda record: record.end_frame)

            for rec_idx, rec in enumerate(vrecords):
                nextclipfile = osp.join(self.data_prefix['video'], rec.next_clip_name)

                # check if the next clip file exists. Since this is frame based, iterate through frames and check if the frame exists
                futr_clip_meta = rec.next_clip_name.split('_')
                futr_clip_start = int(futr_clip_meta[1].split('ps')[-1])
                futr_clip_end = int(futr_clip_meta[2].split('pe')[-1])
                video_name = futr_clip_meta[0]
                video_dir = osp.join(self.data_prefix['video'], "frames", video_name)
                for frame_idx in range(futr_clip_start, futr_clip_end+1):
                    frame_file = osp.join(video_dir, f"{frame_idx:05d}.jpg")
                    if not exists(frame_file):
                        missing_videos += 1
                        print(f"Missing video clip: {nextclipfile}")
                        continue

                cur_clip_steps = np.array(rec.cur_verb)[None, :]
                futr_clip_steps = np.array(rec.futr_verb)[None, :]

                # merge cur + fut state to find state change
                cur_futr_steps = np.concatenate([cur_clip_steps, futr_clip_steps], axis=0) 

                sc_ids_v = gen_state_change_labels(cur_futr_steps)
                sc_labels_list_v.extend(sc_ids_v.numpy().tolist()[0])
                state_tracker.update_per_presence(sc_ids_v.numpy().tolist()[0])
                cur_clip_range = np.arange(rec.start_frame, rec.end_frame+1)
                futr_clip_range = np.arange(futr_clip_start, futr_clip_end+1)

                # check if the future clip is continuation of the current clip
                for p in range(len(cur_clip_range)):
                    if futr_clip_range[p] != cur_clip_range[-1] + p + 1:
                        print(f"Warning: Skipping future clip at timestamp {futr_clip_start} in {vid} is not continuation of current clip at {rec.end_frame} with increment {p}")
                        breakpoint()

                clip_record = deepcopy(rec)
                # state change labels 
                clip_record.sc_labels_v   = sc_ids_v
                # clip labels
                clip_record.cur_v    = cur_clip_steps
                clip_record.futr_v   = futr_clip_steps
                clip_record.futr_clip_meta = [futr_clip_start, futr_clip_end]

                # clip metas
                clip_record.anticipation_window = [3]
                clip_record.video_id = vid
                clip_record.clip_meta = [rec.start_frame, rec.end_frame]
                clip_record.next_clip_file = nextclipfile
                records.append(clip_record)

            if self.num_train_videos > 0 and vid_index >= self.num_train_videos:
                break

        sc_cls_weights_v = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_v), y=sc_labels_list_v)
        print(f"Cls wts for state change verb: {list(sc_cls_weights_v)}")
        print(f"# missing videos: {missing_videos}")
        state_tracker.info()
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
            video_info['sc_v']   = vrecord.sc_labels_v
            video_info['cl_v']   = torch.tensor(vrecord.cur_v).float()
            video_info['futr_v']   = torch.tensor(vrecord.futr_v).float()

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

            cur_clip_range = np.arange(vrecord.clip_meta[0], vrecord.clip_meta[1]+1)
            futr_clip_range = np.arange(vrecord.futr_clip_meta[0], vrecord.futr_clip_meta[1]+1)
            
            # check if the future clip is continuation of the current clip
            for p in range(len(cur_clip_range)):
                if futr_clip_range[p] != cur_clip_range[-1] + p + 1:
                    print(f"Warning: Skipping future clip at timestamp {vrecord.futr_clip_meta[0]} in {vrecord.video_id} is not continuation of current clip at {vrecord.clip_meta[1]} with increment {p}")
                    mismatched_frames += 1

            if self.task  == 'phase-pretraining':
                video_info['label']  = video_info['phase']
            elif self.task == 'verb-pretraining':
                video_info['label']  = video_info['cl_v']
            elif self.task == 'state-change':
                video_info['label']  = video_info['sc_v']
            elif self.task == 'rsd':
                video_info['label']  = torch.tensor(vrecord.rsd_label).unsqueeze(0)
            elif self.task == 'phase-transition':
                video_info['label']  = torch.tensor(vrecord.phtr_label).unsqueeze(0)

            data_list.append(video_info)
        print(f"Total number of samples in dataset: {len(data_list)}")
        print(f"Total number of mismatched frames: {mismatched_frames}")
        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        return data_info