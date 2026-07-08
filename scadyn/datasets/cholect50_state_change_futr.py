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

from .utils import (to_components, gen_state_change_labels, ClipRecord, 
                    SimpleClipRecord, StateTracker, reverse_map)
from sklearn.utils.class_weight import compute_class_weight
bile_duct_triplet_ids = [68, 69, 70, 78, 79, 80, 81]

@DATASETS.register_module()
class CholecStateChangePretrainingDataset(VideoDataset):
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
                clip_record = SimpleClipRecord(clipdata, dur=self.clip_dur)
                records.append(clip_record)
                records_by_vid.setdefault(clip_record.video, []).append(clip_record)
                total_clips += 1
        print(f"Processing # videos {len(records_by_vid)} with # clips {total_clips}...")
        
        records = []
        state_tracker = StateTracker()
        unique_phases = set()
        total_clips, missing_videos, sc_labels_list_v, sc_labels_list_ivt, sc_labels_list_i = 0, 0, [], [], []
        sc_v_matrix = []
        sc_labels_list_iv, sc_labels_list_it = [], []
        phase_labels_list = []
        num_phase_classes = None

        for vid_index, (vid, vrecords) in enumerate(records_by_vid.items()):
            vrecords = sorted(vrecords, key=lambda record: record.end_frame)

            for rec_idx, rec in enumerate(vrecords):     
                nextclipfile = osp.join(self.data_prefix['video'], rec.next_clip_name+".mp4")
                if not exists(nextclipfile):
                    missing_videos += 1
                    print(f"Missing video clip: {nextclipfile}")
                    continue

                cur_clip_ivt  = np.array(rec.cur_triplet)[None, :]
                cur_clip_v    = np.array(rec.cur_verb)[None, :]
                futr_clip_ivt = np.array(rec.futr_triplet)[None, :] # already maxed across time, t=3sec
                futr_clip_v   = to_components(futr_clip_ivt, component=['v']) # type: ignore

                # get it and iv combination 
                cur_clip_it = to_components(cur_clip_ivt, component=['it'])
                futr_clip_it = to_components(futr_clip_ivt, component=['it'])

                cur_clip_iv = to_components(cur_clip_ivt, component=['iv'])
                futr_clip_iv = to_components(futr_clip_ivt, component=['iv'])
                
                assert np.all(futr_clip_v == rec.futr_verb) # sanity check for future verb

                # merge cur + fut state to find state change
                cur_futr_ivt  = np.concatenate([cur_clip_ivt, futr_clip_ivt], axis=0) 
                cur_futr_v    = np.concatenate([cur_clip_v, futr_clip_v], axis=0)

                sc_ids_v = gen_state_change_labels(cur_futr_v)
                sc_ids_ivt = gen_state_change_labels(cur_futr_ivt)
                sc_labels_list_v.extend(sc_ids_v.numpy().tolist()[0])
                sc_labels_list_ivt.extend(sc_ids_ivt.numpy().tolist()[0])
                state_tracker.update_per_presence(sc_ids_v.numpy().tolist()[0])
                sc_v_matrix.append(sc_ids_v.numpy().tolist()[0])

                # extract instrument labels from the triplet labels and compute the state change labels
                cur_clip_i  = to_components(cur_clip_ivt, component=['i'])
                futr_clip_i = to_components(futr_clip_ivt, component=['i'])
                cur_futr_i  = np.concatenate([cur_clip_i, futr_clip_i], axis=0)
                sc_ids_i    = gen_state_change_labels(cur_futr_i)
                sc_labels_list_i.extend(sc_ids_i.numpy().tolist()[0])

                # get sc for iv and iv combinations
                cur_futr_iv = np.concatenate([cur_clip_iv, futr_clip_iv], axis=0)
                sc_ids_iv = gen_state_change_labels(cur_futr_iv)
                sc_labels_list_iv.extend(sc_ids_iv.numpy().tolist()[0])

                cur_futr_it = np.concatenate([cur_clip_it, futr_clip_it], axis=0)
                sc_ids_it = gen_state_change_labels(cur_futr_it)      
                sc_labels_list_it.extend(sc_ids_it.numpy().tolist()[0])

                clip_record = deepcopy(rec)
                # state change labels 
                clip_record.sc_labels_v   = sc_ids_v
                clip_record.sc_labels_ivt = sc_ids_ivt
                clip_record.sc_labels_i   = sc_ids_i
                clip_record.sc_labels_it  = sc_ids_it
                clip_record.sc_labels_iv  = sc_ids_iv

                # clip labels
                clip_record.cur_ivt  = cur_clip_ivt
                clip_record.cur_v    = cur_clip_v
                clip_record.futr_ivt = futr_clip_ivt
                clip_record.futr_v   = futr_clip_v
                clip_record.cur_iv   = cur_clip_iv
                clip_record.cur_it   = cur_clip_it

                clip_record.futr_iv  = futr_clip_iv
                clip_record.futr_it  = futr_clip_it

                # clip metas
                clip_record.anticipation_window = [3]
                clip_record.video_id = [int(vid.split('VID')[-1])]
                clip_record.clip_meta = [rec.start_frame, rec.end_frame]
                clip_record.next_clip_file = nextclipfile
                phase_label = np.argmax(rec.cur_phase)
                if num_phase_classes is None:
                    num_phase_classes = int(len(rec.cur_phase))
                unique_phases.add(phase_label)
                phase_labels_list.append(int(phase_label))
                clip_record.phase = phase_label
                
                # based on task add more labels, tasks include state-change, rsd, and phase-transition
                if self.task == 'rsd':
                    if rec.rsd['disable_clip']:
                        continue
                    clip_record.rsd_label = rec.rsd['value']
                    assert rec.rsd['value'] > 0
                elif self.task == 'phase-transition':
                    if rec.phtr['disable_clip']:
                        continue
                    all_phase_transitions_in_sec = rec.phtr['phase_all_slots']
                    # check if the phase_transition_labels are not less than 0
                    if np.any(np.array(all_phase_transitions_in_sec) < 0):
                        breakpoint()
                        raise ValueError("Phase transition labels are less than 0")
                    
                    if len(all_phase_transitions_in_sec) == 0:
                        print(f"No phase transitions found for {vid}")
                    clip_record.phtr_label = all_phase_transitions_in_sec
                records.append(clip_record)

            if self.num_train_videos > 0 and vid_index >= self.num_train_videos:
                break
        
        sc_cls_weights_v = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_v), y=sc_labels_list_v)
        sc_cls_weights_ivt = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_ivt), y=sc_labels_list_ivt)
        sc_cls_weights_i = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_i), y=sc_labels_list_i)
        sc_cls_weights_iv = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_iv), y=sc_labels_list_iv)
        sc_cls_weights_it = compute_class_weight(class_weight='balanced', classes=np.unique(sc_labels_list_it), y=sc_labels_list_it)
        print(f"Cls wts for state change verb: {list(sc_cls_weights_v)}")
        print(f"Cls wts for state change triplet: {list(sc_cls_weights_ivt)}")
        print(f"Cls wts for state change instrument: {list(sc_cls_weights_i)}")
        print(f"Cls wts for state change instrument-verb: {list(sc_cls_weights_iv)}")
        print(f"Cls wts for state change instrument-target: {list(sc_cls_weights_it)}")
        print(f"# missing videos: {missing_videos}")
        state_tracker.info()
        print(f"Using # videos {len(records)} for {self.split} split...")
        print(f"Unique phases: {unique_phases}")

        # phase label distribution + balanced class weights (for phase-pretraining head)
        if len(phase_labels_list) > 0:
            if num_phase_classes is None:
                num_phase_classes = int(max(phase_labels_list)) + 1
            phase_labels = np.array(phase_labels_list, dtype=np.int64)
            phase_counts = np.bincount(phase_labels, minlength=num_phase_classes).astype(np.float32)
            phase_freq = phase_counts / max(float(phase_counts.sum()), 1.0)

            phase_classes = np.arange(num_phase_classes)
            if np.any(phase_counts == 0):
                # keep finite weights when one class is missing in this split
                safe_counts = np.clip(phase_counts, 1.0, None)
                phase_cls_weights = float(phase_counts.sum()) / (float(num_phase_classes) * safe_counts)
            else:
                phase_cls_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=phase_classes,
                    y=phase_labels)

            self.phase_label_distribution = phase_counts.tolist()
            self.phase_label_frequency = phase_freq.tolist()
            self.phase_cls_weights = phase_cls_weights.tolist()

            print(f"Phase label counts (class-id order): {self.phase_label_distribution}")
            print(f"Phase label frequency (class-id order): {self.phase_label_frequency}")
            print(f"Cls wts for phase labels: {self.phase_cls_weights}")
            print(f"COPY_PHASE_CLASS_WEIGHTS = {self.phase_cls_weights}")

        # get per class per state weights for v2 state change class head
        sc_v_matrix = np.array(sc_v_matrix)
        count_sc_v_matrix = np.stack([np.bincount(sc_v_matrix[:, c], minlength=4) for c in range(sc_v_matrix.shape[1])], axis=0)
        sc_v_weights = count_sc_v_matrix.sum(axis=1, keepdims=True) / np.clip(count_sc_v_matrix, 1, None)
        sc_v_weights = sc_v_weights / sc_v_weights.mean(axis=1, keepdims=True)
        print(f"State change weights per class per state for verb component: {sc_v_weights.tolist()}")

        return records

    def parse_records(self, vrecord):
        all_frame_ids = np.arange(0, self.clip_dur*self.sampling_fps)

        if self.frame_sampling_mode == 'uniform':
            current_frame_ids = all_frame_ids[::self.frame_interval][:self.clip_len]
            future_frame_ids = all_frame_ids[::self.frame_interval][:self.clip_len]

            # pad the last frame if needed
            if len(current_frame_ids) < self.clip_len:
                last_frame = current_frame_ids[-1]
                current_frame_ids = np.pad(current_frame_ids, 
                    (0, self.clip_len - len(current_frame_ids)),
                    mode='constant',
                    constant_values=last_frame)
        elif self.frame_sampling_mode == 'random': # random sampling
            current_frame_ids = np.random.choice(all_frame_ids, self.clip_len, replace=False)
            future_frame_ids = np.random.choice(all_frame_ids, self.clip_len, replace=False)

        return current_frame_ids, future_frame_ids
        
    def load_data_list(self) -> List[dict]:
        exists(self.ann_file)
        self.data = self.load_annotations()
        data_list = []
        for vrecord in self.data:
            current_frame_ids, future_frame_ids = self.parse_records(vrecord)

            video_info = dict()
            video_info['sc_v']   = vrecord.sc_labels_v
            video_info['sc_ivt'] = vrecord.sc_labels_ivt
            video_info['sc_i']   = vrecord.sc_labels_i
            video_info['sc_iv'] = vrecord.sc_labels_iv
            video_info['sc_it'] = vrecord.sc_labels_it
            video_info['cl_v']   = torch.tensor(vrecord.cur_v).float()
            video_info['cl_ivt'] = torch.tensor(vrecord.cur_ivt).float()
            video_info['cl_iv'] = torch.tensor(vrecord.cur_iv).float()
            video_info['cl_it'] = torch.tensor(vrecord.cur_it).float()
            video_info['futr_cl_v']   = torch.tensor(vrecord.futr_v).float()
            video_info['futr_v']   = torch.tensor(vrecord.futr_v).float()
            video_info['futr_ivt'] = torch.tensor(vrecord.futr_ivt).float()
            video_info['futr_iv'] = torch.tensor(vrecord.futr_iv).float()
            video_info['futr_it'] = torch.tensor(vrecord.futr_it).float()

            # metas
            video_info['video_id']  = torch.tensor(vrecord.video_id)
            video_info['clip_meta'] = torch.tensor(vrecord.clip_meta, dtype=torch.float32).unsqueeze(0)
            video_info['anticipation_window'] = torch.tensor(vrecord.anticipation_window)
            video_info['filename']  = osp.join(self.data_prefix['video'], vrecord.clipname)
            video_info['next_clip_filename'] = vrecord.next_clip_file
            video_info['img_shape'] = self.img_shape
            video_info['task'] = self.task
            video_info['frame_inds']   = torch.tensor([int(p) for p in current_frame_ids], dtype=torch.int32)
            video_info['future_frame_inds'] = torch.tensor([int(p) for p in future_frame_ids], dtype=torch.int32)
            video_info['phase'] = torch.tensor([vrecord.phase]).long()

            if self.task  == 'phase-pretraining':
                video_info['label']  = video_info['phase']
            elif self.task == 'verb-pretraining':
                video_info['label']  = video_info['cl_v']
            elif self.task == 'state-change':
                video_info['label']  = video_info['sc_v']
            elif self.task == 'triplet-state-change':
                video_info['label']  = video_info['sc_ivt']
            elif self.task == 'instrument-state-change':
                video_info['label']  = video_info['sc_i']
            elif self.task == 'instrument-verb-state-change':
                video_info['label']  = video_info['sc_iv']  
            elif self.task == 'instrument-target-state-change':
                video_info['label']  = video_info['sc_it']              
            elif self.task == 'rsd':
                video_info['label']  = torch.tensor(vrecord.rsd_label).unsqueeze(0)
            elif self.task == 'phase-transition':
                video_info['label']  = torch.tensor(vrecord.phtr_label).unsqueeze(0)
            elif self.task == 'instrument-anticipation':
                video_info['label']  = torch.tensor(vrecord.iantr_label).unsqueeze(0)

            data_list.append(video_info)
        print(f"Total number of samples in dataset: {len(data_list)}")
        return data_list
    
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index
        return data_info
