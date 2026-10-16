"""
From the GraSP dataset with frame level phase and step labels, get the anticipation labels
1. RSD style
2. Phase transition style
3. Step transition style
"""

import os
import json
import numpy as np
from collections import Counter

train_video_list = ['CASE001', 'CASE002', 'CASE003', 'CASE004', 'CASE014', 'CASE015', 'CASE021']
val_video_list = ['CASE007']
test_video_list = ['CASE041', 'CASE047', 'CASE050', 'CASE051', 'CASE053']

video_mode_dict = {
    'train': train_video_list,
    'val': val_video_list,
    'test': test_video_list
}

from copy import deepcopy
def get_next_label_transition(start_t, 
                              end_t, 
                              label_data
                              ):
    # get the number of classes
    num_classes = label_data.shape[1]

    # get the phase for the current clip majority vote
    event_data = []
    for frame_idx in range(start_t, end_t+1): # end_t is inclusive so added 1 NOTE
        frame_data = label_data[frame_idx]
        event_label = np.where(frame_data == 1)[0][0]
        event_data.append(event_label)
    event = Counter(event_data).most_common(1)[0][0]
    current_event_label = deepcopy(event)

    # look for next event transition (seconds and the class label)
    # NOTE: ignore idle phase and step. E.g. phase 8 occurs then moves to 0 and back to 8
    next_event_time_in_sec = []
    num_next_event_transitions = 0
    for frame_idx in range(end_t+1, label_data.shape[0]):
        frame_data = label_data[frame_idx]
        event_label = np.where(frame_data == 1)[0][0]
        if event_label != event and event_label != 0: # 0 is idle phase and step
            next_event_time_in_sec.append({
                'time_in_sec': frame_idx - end_t,
                'next_event_label': event_label
            })
            num_next_event_transitions += 1
            event = event_label

    if num_next_event_transitions == 0:
        return []
    
    # now compute phase transition labels for the next 2, 3, 5 minutes
    time2next_event = next_event_time_in_sec[0]['time_in_sec'] # in seconds
    time2next_event_min = time2next_event / 60.0
    next_event_label = next_event_time_in_sec[0]['next_event_label']
 
    event_all_slots = []
    futr_time_slots = [2, 3, 5] # in min

    # convert next_event_time_in_sec to min 
    # NOTE: this does it for all classes as some classes occur immediately after the next class
    next_event_time_in_min = [x['time_in_sec'] / 60.0 for x in next_event_time_in_sec]
    for ne in futr_time_slots:
        event_onehot = [ne] * num_classes
        event_onehot[current_event_label] = 0
        next_starting_event = current_event_label + 1
        next_ending_event = next_starting_event + 1
        
        # ignore the last event or class
        if next_ending_event > num_classes:
            continue
        
        for i in range(next_starting_event, next_ending_event):
            try:
                time_to_next_event = next_event_time_in_min[i-current_event_label-1]
            except:
                breakpoint()
                #raise ValueError(f"Index out of bounds for the clip with start time {start_t} and end time {end_t}")
                # print(f"Index out of bounds for the clip with start time {start_t} and end time {end_t}")
                continue
            if time_to_next_event <= ne:
                try:
                    event_onehot[i] = round(time_to_next_event, 4)
                except:
                    breakpoint()
                    raise ValueError(f"Illegal index")
            else:
                break
        
        # check if the event is not less than 0
        if np.any(np.array(event_onehot) < 0):
            breakpoint()
            raise ValueError(f"Event is less than 0")
        
        event_all_slots.extend(event_onehot)

    return event_all_slots

def calculate_anticipation_labels(clip_metadata, frame_stride=60, clip_duration=4):
    """
    Calculate the anticipation labels for the given clip metadata
    """
    modes = ['train', 'val', 'test']
    all_antr_data = {}
    for mode in modes:
        all_antr_data[mode] = {}
    
    for mode in modes:
        video_antr_data = {}
        mode_video_list = video_mode_dict[mode]
        for video_name, video_data in clip_metadata.items():
            if video_name not in mode_video_list:
                continue
            
            step_labels = video_data['label_matrix']
            phase_labels = video_data['phase_label_matrix']
            total_frames = step_labels.shape[0]

            start_frame = video_data['frame_info']['start_frame']
            end_frame = video_data['frame_info']['end_frame']
            num_frames = end_frame - start_frame + 1
            num_anticipation_frames = num_frames // frame_stride
            # TODO: add the logic to calculate the anticipation labels

            if video_name not in video_antr_data:
                video_antr_data[video_name] = []

            # to ignore the last 100 frames in case any step transition occurs
            # and start 100 frames from the start of the video
            for clip_start_t in range(start_frame+100, end_frame-100, frame_stride):
                clip_end_t = clip_start_t + clip_duration - 1

                cur_clip_id = f"{video_name}_ps{clip_start_t}_pe{clip_end_t}"
                futr_clip_start_t = clip_end_t+1
                futr_clip_end_t = futr_clip_start_t + clip_duration -1
                futr_clip_id = f"{video_name}_ps{futr_clip_start_t}_pe{futr_clip_end_t}"
                
                # get rsd
                rsd = total_frames - clip_end_t - 1
                rsd = round(rsd / (60.0 * 5.0), 2)

                # get the phase transition labels on current clip
                phase_transition_labels = get_next_label_transition(clip_start_t, 
                                                                        clip_end_t, 
                                                                        phase_labels)
                if len(phase_transition_labels) == 0:
                    continue
                
                # get the step transition labels on current clip
                step_transition_labels = get_next_label_transition(clip_start_t, 
                                                                        clip_end_t, 
                                                                        step_labels)
                if len(step_transition_labels) == 0:
                    continue

                # print(step_transition_labels)
                # print(phase_transition_labels)
                
                # create the clip level metadata
                clip_meta = {
                    'clip_id': cur_clip_id,
                    'start_t': clip_start_t,
                    'end_t': clip_end_t,
                    'cur_clip_name': cur_clip_id,
                    'futr_clip_name': futr_clip_id,
                    'futr_start_t': futr_clip_start_t,
                    'futr_end_t': futr_clip_end_t,
                    'rsd': rsd,
                    'phase_transition': phase_transition_labels,
                    'step_transition': step_transition_labels
                }

                # add the clip metadata to the video metadata
                video_antr_data[video_name].append(clip_meta)

        # count number of clips per video
        total_clips = 0
        for video_name, video_data in video_antr_data.items():
            total_clips += len(video_data)
        print(f"Total number of clips in {mode}: {total_clips}")
        all_antr_data[mode] = video_antr_data
    return all_antr_data

def get_ann_meta(label_files):
    num_step_classes = 21 # 21 steps
    num_phase_classes = 11

    all_videos = {}
    video_labels = {}
    phase_classes = set()
    for e, label_file in enumerate(label_files):
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        cats = data['steps_categories']
        images = data['images']
        annotations = data['annotations']

        # go through the annotations and save image and annotation label per video
        for ann in annotations:
            # breakpoint()
            frame_id = ann['image_name'] # CASE001/00000.jpg
            video_name = frame_id.split('/')[0]

            # check how many phases are there for each video
            phase_classes.add(ann['phases'])

            step_id = ann['steps']
            if video_name not in video_labels:
                video_labels[video_name] = {}
                video_labels[video_name]['gt'] = []
            
            ann_meta = {
                'video_name': video_name,
                'frame_id': frame_id,
                'label': step_id,
                'step_label': step_id,
                'phase_label': ann['phases']
            }
            video_labels[video_name]['gt'].append(ann_meta)
    
    for video_name, video_data in video_labels.items():
        gt = video_data['gt']
        start_frame = int(gt[0]['frame_id'].split('/')[-1].split('.')[0])
        end_frame = int(gt[-1]['frame_id'].split('/')[-1].split('.')[0])
        video_labels_onehot = []
        video_phase_labels_onehot = []
        for i, ann in enumerate(gt):
            frame_id = ann['frame_id']
            subfolder = 'grasp' # 'GraSP_1fps'
            frame_file = os.path.join('videos', subfolder, 'frames',frame_id)
            if not os.path.exists(frame_file):
                print(f"Missing frame {frame_file} for video {video_name}")
                continue
            frame_label = ann['label']
            label_one_hot = np.zeros(num_step_classes)
            phase_one_hot = np.zeros(num_phase_classes)

            try:
                label_one_hot[int(frame_label)] = 1
                phase_one_hot[int(ann['phase_label'])] = 1
            except:
                breakpoint()
                print(f'Warning: Label {frame_label} not found in one hot encoding')

            video_labels_onehot.append(label_one_hot)
            video_phase_labels_onehot.append(phase_one_hot)

        video_labels_onehot = np.array(video_labels_onehot)
        video_phase_labels_onehot = np.array(video_phase_labels_onehot)
        all_videos[video_name] = {}
        all_videos[video_name]['label_matrix'] = video_labels_onehot
        all_videos[video_name]['phase_label_matrix'] = video_phase_labels_onehot
        all_videos[video_name]['frame_info'] = {'start_frame': start_frame, 'end_frame': end_frame}
    return all_videos

if __name__ == '__main__':
    path = "<YOUR_PATH>/GraSP_1fps"
    label_files = [os.path.join(path, "annotations", "grasp_long-term_train.json"), os.path.join(path, "annotations", "grasp_long-term_test.json")]
    clip_duration = 4
    frame_stride = 20 # in seconds on frames that are extracted at 1 fps

    all_videos = get_ann_meta(label_files)
    
    anticipation_labels = calculate_anticipation_labels(all_videos, frame_stride=frame_stride, clip_duration=clip_duration)
    out_data = {}
    out_data['anticipation_labels'] = anticipation_labels
    out_data['clip_duration'] = clip_duration
    out_data['frame_stride'] = frame_stride
    
    # NOTE: Uncomment below to save the anticipation labels to a json file
    # USED only for RSD, Phase and Step transition style anticipation labels. For Step Anticipation classification style, use the get_anticipation_labels.py script
    # #train: 2802, #val:341, #test:1718
    
    # out_file = f'data/rarp_state_change/frame{clip_duration}s_grasp_anticipation_v1.json'
    # with open(out_file, 'w') as f:
    #     json.dump(out_data, f, indent=2)
    breakpoint()