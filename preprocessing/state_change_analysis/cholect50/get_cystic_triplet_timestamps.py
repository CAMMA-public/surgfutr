import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from copy import deepcopy
from preprocessing.state_change_analysis.cholect50.misc import (check_overlap, 
            find_stable_transitions_with_label, get_label_onehot)

label_path = '<YOUR_PATH>/CholecT50/labels'
label_files = glob.glob(os.path.join(label_path, '*.json'))

triplet_ids_selected = [58, 59, 57, 79, 78, 69, 68, 84, 34]
triplet_id2name = {}

# load ct50 dict name mapping 
name_mapping = pd.read_csv(os.path.join('<YOUR_PATH>/CholecT50/dict/triplet.txt'), header=None).to_dict(orient='records')
for name in name_mapping:
    tool_name = name[0]
    verb_name = name[1]
    target_name = name[2]
    triplet_index, tool_name = tool_name.split(':')
    
    if 'cystic' in target_name:
        triplet_id2name[triplet_index] = ','.join([tool_name, verb_name, target_name])

# train, val, test - rdv split
train_video_list = ['VID01', 'VID15', 'VID26', 'VID40', 'VID52', 'VID65', 'VID79', 'VID02', 'VID18', 'VID27', 'VID43', 'VID56', 'VID66', 'VID92', 
                    'VID04', 'VID22', 'VID31', 'VID47', 'VID57', 'VID68', 'VID96', 'VID05', 'VID23', 'VID35', 'VID48', 'VID60', 'VID70', 'VID103', 
                    'VID13', 'VID25', 'VID36', 'VID49', 'VID62', 'VID75', 'VID110']
val_video_list = ['VID08', 'VID12', 'VID29', 'VID50', 'VID78']
test_video_list = ['VID06', 'VID51', 'VID10', 'VID73', 'VID14', 'VID74', 'VID32', 'VID80', 'VID42', 'VID111']

def generate_clip_metadata(all_transitions,
                           train_video_list, 
                           val_video_list, 
                           test_video_list, 
                           verb_onehot,
                           triplet_onehot,
                           phase_onehot,
                           clip_duration=3,
                           anticipation_durations=[1, 3, 5]):
    clip_metadata = {'train': {}, 'val': {}, 'test': {}}
    
    for video_name in all_transitions.keys():
        if video_name in train_video_list:
            split = 'train'
        elif video_name in val_video_list:
            split = 'val'
        elif video_name in test_video_list:
            split = 'test'
        else:
            print(f"Warning: Video {video_name} not found in any split. Skipping.")
            continue

        transitions = all_transitions.get(video_name, [])
        video_labels = verb_onehot.get(video_name, [])
        video_triplet_labels = triplet_onehot.get(video_name, [])
        video_phase_labels = phase_onehot.get(video_name, [])
        
        num_frames, _ = video_labels.shape
        all_timestamps = transitions

        clip_list = []
        video_clips = []
        for (t, triplet_id) in all_timestamps: # all timestamps denote where the change happens
            # get the triplet id that appears at this timestamp
            for anticipation_duration in anticipation_durations:
                start_t = max(0, t - anticipation_duration - clip_duration)
                end_t = t-anticipation_duration

                # Check if this clip would be too short due to overlap adjustment
                if end_t - start_t < clip_duration:
                    print(f"Warning: Skipping clip at timestamp {t} in {video_name} due to overlap")
                    continue
                
                clip_id = f"{video_name}_ps{start_t}_pe{end_t}"

                # to check the new clips overlaps with the existing clips
                clip_list.append({
                    "clip_id": clip_id,
                    "start_t": start_t,
                    "end_t": end_t,
                })

                # get current clip triplet, phase and verb labels
                cur_clip_triplet_label = video_triplet_labels[start_t:end_t+1].max(axis=0)
                cur_clip_phase_label = video_phase_labels[start_t:end_t+1].max(axis=0)
                cur_clip_verb_label = video_labels[start_t:end_t+1].max(axis=0)
                
                video_clips.append({
                    "clip_id": clip_id,
                    "start_t": start_t,
                    "end_t": end_t,
                    "cur_clip_name": clip_id,
                    "event_t": t,
                    "event_triplet_id":triplet_id,
                    "anticipation_duration": anticipation_duration,
                    "cur_verb": cur_clip_verb_label.tolist(),
                    "cur_triplet": cur_clip_triplet_label.tolist(),
                    "cur_phase": cur_clip_phase_label.tolist(),
                })

        clip_metadata[split][video_name] = video_clips
    
    return clip_metadata

if __name__ == '__main__':
    triplet_onehot = get_label_onehot(mode='triplet', label_files=label_files)
    phase_onehot = get_label_onehot(mode='phase', label_files=label_files)
    verb_onehot = get_label_onehot(mode='verb', label_files=label_files)

    all_onset_transitions = {}
    for video_name, video_triplet_labels in triplet_onehot.items():
        all_onset_transitions[video_name] = find_stable_transitions_with_label(video_triplet_labels, src_label=0, dst_label=1, k=3, overlap_threshold=1, filter_classes=triplet_ids_selected)
    print(f'Total transitions: {sum(len(t) for t in all_onset_transitions.values())}')

    # transition timestamps per split from all_onset_transitions
    train_cystic_triplets_transitions = {k: v for k, v in all_onset_transitions.items() if k in train_video_list}
    total_train_transitions = sum(len(t) for t in train_cystic_triplets_transitions.values())
    val_cystic_triplets_transitions = {k: v for k, v in all_onset_transitions.items() if k in val_video_list}
    total_val_transitions = sum(len(t) for t in val_cystic_triplets_transitions.values())
    test_cystic_triplets_transitions = {k: v for k, v in all_onset_transitions.items() if k in test_video_list}
    total_test_transitions = sum(len(t) for t in test_cystic_triplets_transitions.values())

    print(f'Total train transitions: {total_train_transitions}, Total val transitions: {total_val_transitions}, Total test transitions: {total_test_transitions}')

    clip_metadata = generate_clip_metadata(all_onset_transitions,
                           train_video_list, 
                           val_video_list, 
                           test_video_list, 
                           verb_onehot,
                           triplet_onehot,
                           phase_onehot,
                           clip_duration=3,
                           anticipation_durations=[1, 3, 5, 7])
    
    # NOTE: Uncomment below to save the clip metadata before extracting clips
    #out_file = f'data/hct_triplets/clip3s_HCT_triplets_anticipation_v1.json'
    #with open(out_file, 'w') as f:
    #    json.dump(clip_metadata, f, indent=2)

    # Print overall statistics
    total_clips = sum(sum(len(clips) for clips in split_data.values()) for split_data in clip_metadata.values())
    print("\nOverall statistics:")
    print(f"Total number of clips across all splits: {total_clips}")
    breakpoint()