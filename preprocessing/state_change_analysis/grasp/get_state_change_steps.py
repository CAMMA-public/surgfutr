"""
get label transition from the GraSP dataset using steps
"""

import os
import json
import numpy as np
from tqdm import tqdm
import glob
from copy import deepcopy
from misc import check_overlap, find_stable_transitions, get_label_onehot_steps

# train, val, test - rdv split
train_video_list = ['CASE001', 'CASE002', 'CASE003', 'CASE004', 'CASE014', 'CASE015', 'CASE021']
val_video_list = ['CASE007']
test_video_list = ['CASE041', 'CASE047', 'CASE050', 'CASE051', 'CASE052', 'CASE053']

def get_continuity_timestamps(verb_onehot, all_transitions, label_value=1, k=3, window_size=20):
    all_continuity_timestamps = {}
    
    for video_name, video_verb_labels in verb_onehot.items():
        num_frames, num_classes = video_verb_labels['label_matrix'].shape
        label_matrix = video_verb_labels['label_matrix']
        frame_info = video_verb_labels['frame_info']
        start_frame = frame_info['start_frame']
        end_frame = frame_info['end_frame']

        transitions = set(all_transitions.get(video_name, []))
        continuity_timestamps = []

        for window_start in range(start_frame, end_frame, window_size):
            window_end = min(window_start + window_size, num_frames)
            
            window_continuity_found = False
            for frame in range(window_start + k, window_end - k):
                if any(abs(frame - t) <= k for t in transitions):
                    continue  # Skip frames near transitions
                
                valid_idx = [frame + i for i in range(-k, k+1)]
                if not all(i >= 0 and i < num_frames for i in valid_idx):
                    continue
                
                frame_offset = frame - start_frame
                for cls in range(num_classes):
                    # Check for continuity of label_value
                    if (label_matrix[frame_offset, cls] == label_value and
                        all(label_matrix[frame_offset + i, cls] == label_value for i in range(-k, k+1))):
                        continuity_timestamps.append(frame)
                        window_continuity_found = True
                        break  # Move to next window after finding continuity in any class
                
                if window_continuity_found:
                    break  # Move to next window

        all_continuity_timestamps[video_name] = continuity_timestamps

    # count the number of stable timestamps per video and total stable timestamps
    total_continuity_timestamps = 0
    for video_name, continuity_timestamps in all_continuity_timestamps.items():
        total_continuity_timestamps += len(continuity_timestamps)
        # print(f'{video_name}: {len(continuity_timestamps)} continuity timestamps')
    print(f'Total continuity timestamps for {label_value}: {total_continuity_timestamps}')
    return all_continuity_timestamps

def generate_clip_metadata(all_transitions, 
                           all_continuity_timestamps_1, 
                           all_continuity_timestamps_0, 
                           train_video_list, 
                           val_video_list, 
                           test_video_list, 
                           verb_onehot=None,
                           triplet_onehot=None,
                           phase_onehot=None,
                           clip_duration=3):
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
        continuity_1 = all_continuity_timestamps_1.get(video_name, [])
        continuity_0 = all_continuity_timestamps_0.get(video_name, [])
        video_labels = verb_onehot.get(video_name, [])
        if triplet_onehot is not None:
            video_triplet_labels = triplet_onehot.get(video_name, [])
        else:
            video_triplet_labels = None
        if phase_onehot is not None:
            video_phase_labels = phase_onehot.get(video_name, [])
        
        num_frames, _ = video_labels['label_matrix'].shape
        label_matrix = video_labels['label_matrix']
        frame_info = video_labels['frame_info']
        start_frame = frame_info['start_frame']
        end_frame = frame_info['end_frame']

        all_timestamps = sorted(set(transitions + continuity_1 + continuity_0))

        clip_list = []
        
        video_clips = []
        for t in all_timestamps: # all timestamps denote where the change happens
            # start_t = max(0, t - clip_duration-1)
            start_t = max(0, t - clip_duration) # NOTE: for frame based state-change
            end_t = t-1

            # Check if this clip would be too short due to overlap adjustment
            if (end_t - start_t + 1) < clip_duration:
                print(f"Warning: Skipping clip at timestamp {t} in {video_name} due to invalid clip length")
                continue

            # check if it overlaps with every clip in clip_list
            for clip in clip_list:
                if start_t <= clip['end_t'] and end_t >= clip['start_t']:
                    print(f"Warning: Skipping clip at timestamp {t} in {video_name} due to overlap")
                    continue
            
            clip_id = f"{video_name}_ps{start_t}_pe{end_t}"

            # to check the new clips overlaps with the existing clips
            clip_list.append({
                "clip_id": clip_id,
                "start_t": start_t,
                "end_t": end_t,
            })

            # PREPARE LABELS TO SAVE IT IN THE JSON FILE
            # get the over label from start_t to end_t with argmax over one hot encoded verb labels
            # cur_clip_label = video_labels[start_t:end_t+1].max(axis=0)

            # NOTE: for frame based state-change
            start_t_offset = start_t - start_frame
            end_t_offset = end_t - start_frame
            cur_clip_label = video_labels['label_matrix'][start_t_offset:end_t_offset+1].max(axis=0)

            # if the current clip is valid then create the future clip meta
            futr_clip_start_t = t
            # futr_clip_end_t = t+clip_duration
            futr_clip_end_t = t+clip_duration-1 # NOTE: for frame based state-change

            # NOTE: for frame based state-change
            futr_clip_start_t_offset = futr_clip_start_t - start_frame
            futr_clip_end_t_offset = futr_clip_end_t - start_frame

            if futr_clip_end_t >= end_frame: # NOTE: for frame based state-change
                print(f"Warning: Skipping future clip at timestamp {t} in {video_name} due to invalid clip length")
                continue

            futr_clip_id = f"{video_name}_ps{futr_clip_start_t}_pe{futr_clip_end_t}"

            cur_clip_range = np.arange(start_t, end_t+1)
            futr_clip_range = np.arange(futr_clip_start_t, futr_clip_end_t+1)

            # check if the future clip is continuation of the current clip
            for p in range(len(cur_clip_range)):
                if futr_clip_range[p] != cur_clip_range[-1] + p + 1:
                    print(f"Warning: Skipping future clip at timestamp {t} in {video_name} is not continuation of current clip at {cur_clip_range[-1]} with increment {p}")
                    breakpoint()

            # get the label for the future clip
            # futr_clip_label = video_labels[futr_clip_start_t:futr_clip_end_t+1].max(axis=0)
            futr_clip_label = video_labels['label_matrix'][futr_clip_start_t_offset:futr_clip_end_t_offset+1].max(axis=0) # NOTE: for frame based state-change

            # for each class, check whether the class label changes from 0 to 1 (onset) or 1 to 0 (discontinuity) or 1 to 1 (continuity) or 0 to 0 (background)
            num_sc_instances = {'onset': 0, 'discontinuity': 0, 'continuity': 0}
            for cls_label in range(len(cur_clip_label)):
                if cur_clip_label[cls_label] == 0 and futr_clip_label[cls_label] == 1:
                    num_sc_instances['onset'] += 1
                elif cur_clip_label[cls_label] == 1 and futr_clip_label[cls_label] == 0:
                    num_sc_instances['discontinuity'] += 1
                elif cur_clip_label[cls_label] == 1 and futr_clip_label[cls_label] == 1:
                    num_sc_instances['continuity'] += 1

            # get current clip triplet and phase labels
            # cur_clip_triplet_label = video_triplet_labels[start_t:end_t+1].max(axis=0)
            if triplet_onehot is not None:
                cur_clip_triplet_label = video_triplet_labels['label_matrix'][start_t_offset:end_t_offset+1].max(axis=0) # NOTE: for frame based state-change
                cur_clip_triplet_label = cur_clip_triplet_label.tolist()
            else:
                cur_clip_triplet_label = None

            if phase_onehot is not None:
                # cur_clip_phase_label = video_phase_labels[start_t:end_t+1].max(axis=0)
                cur_clip_phase_label = video_phase_labels['label_matrix'][start_t_offset:end_t_offset+1].max(axis=0) # NOTE: for frame based state-change

            # future clip triplet and phase labels
            # futr_clip_triplet_label = video_triplet_labels[futr_clip_start_t:futr_clip_end_t+1].max(axis=0)
            if triplet_onehot is not None:
                futr_clip_triplet_label = video_triplet_labels['label_matrix'][futr_clip_start_t_offset:futr_clip_end_t_offset+1].max(axis=0) # NOTE: for frame based state-change
                futr_clip_triplet_label = futr_clip_triplet_label.tolist()
            else:
                futr_clip_triplet_label = None

            if phase_onehot is not None:
                # futr_clip_phase_label = video_phase_labels[futr_clip_start_t:futr_clip_end_t+1].max(axis=0)
                futr_clip_phase_label = video_phase_labels['label_matrix'][futr_clip_start_t_offset:futr_clip_end_t_offset+1].max(axis=0) # NOTE: for frame based state-change

            # find the rsd, phase transition, step transition NOTE: action labels in grasp are on a smaller set and so not used for evaluation
            # if num_sc_instances['onset'] > 0:
            #     breakpoint()
            
            save_dict = {
                "clip_id": clip_id,
                "start_t": start_t,
                "end_t": end_t,
                "cur_clip_name": clip_id,
                "futr_clip_name": futr_clip_id,
                "event_t": t,
                "cur_verb": cur_clip_label.tolist(),
                "cur_triplet": cur_clip_triplet_label,
                "futr_start_t": futr_clip_start_t,
                "futr_end_t": futr_clip_end_t,
                "futr_verb": futr_clip_label.tolist(),
                "futr_triplet": futr_clip_triplet_label,
                "num_onset": 1 if num_sc_instances['onset'] > 0 else 0,
                "num_discontinuity": 1 if num_sc_instances['discontinuity'] > 0 else 0,
                "num_continuity": 1 if num_sc_instances['continuity'] > 0 else 0,
            }

            if phase_onehot is not None:
                save_dict['cur_phase'] = cur_clip_phase_label.tolist()
                save_dict['futr_phase'] = futr_clip_phase_label.tolist()
            video_clips.append(save_dict)

        clip_metadata[split][video_name] = video_clips
    
    return clip_metadata

if __name__ == '__main__':
    path = "<YOUR_PATH>/GraSP_1fps"
    label_files = [os.path.join(path, "annotations", "grasp_long-term_train.json"), os.path.join(path, "annotations", "grasp_long-term_test.json")]

    verb_onehot = get_label_onehot_steps(label_files=label_files)
    clip_duration = 4

    # get the transitions 0 to 1
    all_onset_transitions = {}
    num_skipped_transitions_onset = 0
    for video_name, video_verb_labels in verb_onehot.items():
        all_onset_transitions[video_name], skipped = find_stable_transitions(video_verb_labels, src_label=0, dst_label=1, k=clip_duration, video_name=video_name)
        num_skipped_transitions_onset += skipped
    print(f'# Onset transitions: {sum(len(t) for t in all_onset_transitions.values())}')
    print(f'# Skipped onset transitions: {num_skipped_transitions_onset}')

    # get the transitions 1 to 0
    all_discontinuity_transitions = {}
    num_skipped_transitions_discontinuity = 0
    for video_name, video_verb_labels in verb_onehot.items():
        all_discontinuity_transitions[video_name], skipped = find_stable_transitions(video_verb_labels, src_label=1, dst_label=0, k=clip_duration, transition_list=all_onset_transitions[video_name])
        num_skipped_transitions_discontinuity += skipped
    print(f'# Discontinuity transitions: {sum(len(t) for t in all_discontinuity_transitions.values())}')
    print(f'# Skipped discontinuity transitions: {num_skipped_transitions_discontinuity}')
    # NOTE: the onset timestamps are also the same point for the discontinuity transitions !!!!!!!!!!!!!!!!!!!!!!!!!!!

    # OVERLAP CHECK: find if there are any overlaps between onset and discontinuity transitions ------------
    for video_name, video_verb_labels in verb_onehot.items():
        for t in all_onset_transitions[video_name]:
            # if the range of k frames around the transition contains any discontinuity transition, then there is an overlap
            if any(abs(t - d) <= 3 for d in all_discontinuity_transitions[video_name]):
                print(f'Overlap found between {video_name} onset {t} and discontinuity {all_discontinuity_transitions[video_name]}')
    # ------------------------------------------------------------------------------------------------

    # combine transition from onset and discontinuity for all videos by merging the lists
    all_transitions = {}
    for video_name in verb_onehot.keys():
        all_transitions[video_name] = list(set(all_onset_transitions[video_name] + all_discontinuity_transitions[video_name]))
    print(f'# Combined O-D transitions: {sum(len(t) for t in all_transitions.values())}')

    # get timestamps where verb label is continuous for k frames
    all_continuity_timestamps_1 = get_continuity_timestamps(verb_onehot, all_transitions, label_value=1, k=clip_duration, window_size=90)
    # # #  check for all values of window_size from 0 to 100
    # for window_size in range(1,100):
    #     all_continuity_timestamps_1 = get_continuity_timestamps(verb_onehot, all_transitions, label_value=1, k=clip_duration, window_size=window_size)
    #     print(f'Window size: {window_size}, # Continuity timestamps: {sum(len(t) for t in all_continuity_timestamps_1.values())}')
    # breakpoint()

    # ------------------------------------------------------------------------------------------------
    # OVERLAP CHECK: find if there are any overlaps between continuity of 1 and 0 ------------
    for video_name, continuity_timestamps in all_continuity_timestamps_1.items():
        transitions = all_transitions.get(video_name, [])
        for t in continuity_timestamps:
            if any(abs(t - d) <= 3 for d in transitions):
                print(f'Overlap found in {video_name} between continuity timestamp 1 {t} and transitions {transitions}')
    # ------------------------------------------------------------------------------------------------

    onset_discontinuity_transitions = deepcopy(all_transitions) # only onset and discontinuity transitions

    # update all_transitions with continuity timestamps to use to find continuity of 0
    for video_name, continuity_timestamps in all_continuity_timestamps_1.items():
        all_transitions[video_name] = list(set(all_transitions[video_name] + continuity_timestamps))
    all_continuity_timestamps_0 = get_continuity_timestamps(verb_onehot, all_transitions, label_value=0, k=clip_duration, window_size=50) # continuity of 0

    # ------------------------------------------------------------------------------------------------
    # OVERLAP CHECK: find if there are any overlaps between continuity of 1 and 0 ------------
    for video_name, continuity_timestamps in all_continuity_timestamps_0.items():
        transitions = all_transitions.get(video_name, [])
        for t in continuity_timestamps:
            if any(abs(t - d) <= 3 for d in transitions):
                print(f'Overlap found in {video_name} between continuity timestamp 0 {t} and transitions {transitions}')
    # ------------------------------------------------------------------------------------------------

    # for each video, find the onset, discontinuity, continuity of 1 and continuity of 0 and check if timestamps [k-3, k+3] are overlapping
    check_overlap(verb_onehot, onset_discontinuity_transitions, all_continuity_timestamps_1, all_continuity_timestamps_0)

    # triplet_onehot = get_label_onehot(mode='triplet', label_files=label_files)
    # phase_onehot = get_label_onehot(mode='tool', label_files=label_files)
    triplet_onehot = None
    phase_onehot = None

    clip_metadata = generate_clip_metadata(all_transitions, 
                                           all_continuity_timestamps_1, 
                                           all_continuity_timestamps_0, 
                                           train_video_list, 
                                           val_video_list, 
                                           test_video_list, 
                                           verb_onehot,
                                           triplet_onehot,
                                           phase_onehot,
                                           clip_duration=clip_duration)

    # find distribution % of each type of state change over all clips in each split
    all_split_clips = 0 
    for split in clip_metadata.keys():
        total_onsets = 0
        total_discontinuities = 0
        total_continuities = 0
        total_clips = 0
        for video_name, video_clips in clip_metadata[split].items():
            total_onsets += sum(clip['num_onset'] for clip in video_clips)
            total_discontinuities += sum(clip['num_discontinuity'] for clip in video_clips)
            total_continuities += sum(clip['num_continuity'] for clip in video_clips)
            total_clips += len(video_clips)
        all_split_clips += total_clips

        print(f'{split}: {total_onsets} onsets ({total_onsets/total_clips*100:.2f}%), '
              f'{total_discontinuities} discontinuities ({total_discontinuities/total_clips*100:.2f}%), '
              f'{total_continuities} continuities ({total_continuities/total_clips*100:.2f}%)')
    
    num_train_clips = 0
    num_val_clips = 0
    num_test_clips = 0
    for video_name, clips in clip_metadata["train"].items():
        num_train_clips += len(clips)
    for video_name, clips in clip_metadata["val"].items():
        num_val_clips += len(clips)
    for video_name, clips in clip_metadata["test"].items():
        num_test_clips += len(clips)

    print(f'Total clips: {all_split_clips}; Train: {num_train_clips}; Val: {num_val_clips}; Test: {num_test_clips}')
    # print("check for labels >>> ", clip_metadata['train']['easdv4'][0])
    
    # NOTE: Uncomment below to save the clip metadata to a json file
    # out_file = f'data/rarp_state_change/frame{clip_duration}s_grasp1fps_steps_state_change_v1.json'
    # with open(out_file, 'w') as f:
    #     json.dump(clip_metadata, f, indent=2)

    # Print overall statistics
    total_clips = sum(sum(len(clips) for clips in split_data.values()) for split_data in clip_metadata.values())
    print("\nOverall statistics:")
    print(f"Total number of clips across all splits: {total_clips}")