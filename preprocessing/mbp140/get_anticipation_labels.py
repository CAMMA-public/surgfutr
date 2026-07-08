"""
From the MultiBypass140 dataset with frame level phase and step labels, 
get the anticipation labels :
1. RSD style
2. Phase transition style
3. Step transition style
4. Step Anticipation classification style (1, 3, 5 seconds)
"""

import os
import json
import pickle
import numpy as np
from collections import Counter

stras_train_video_list = ['BP39-10228', 'BP48-7542', 'BP73-13493', 'BP41-10279', 'BP06-7769', 'BP40-10258', 'BP19-12702', 'BP54-12343', 
'BP46-10384', 'BP09-8187', 'BP71-13438', 'BP17-12665', 'BP45-10352', 'BP42-10310', 'BP05-7759', 'BP50-9453', 'BP04-7590', 'BP68-13232', 
'BP01-10156', 'BP62-12739', 'BP51-9561', 'BP12-10113', 'BP25-13689', 'BP11-8427', 'BP29-8453', 'BP53-12335', 'BP20-12926', 'BP31-8547', 
'BP59-12621', 'BP16-12537', 'BP60-12626', 'BP63-12872', 'BP55-12353', 'BP21-13097', 'BP08-8133', 'BP32-8695', 'BP69-13282', 'BP24-13578', 'BP75-13726', 'BP70-13318']
stras_val_video_list = ['BP22-13117', 'BP56-12474', 'BP36-10128', 'BP67-13202', 'BP61-12630', 'BP49-8263', 'BP38-10186', 'BP66-13153', 'BP10-8192', 'BP47-7423']
stras_test_video_list = ['BP23-13142', 'BP52-9693', 'BP18-12676', 'BP27-8207', 'BP74-13718', 'BP43-10336', 'BP72-13491', 'BP14-12440', 'BP57-12516', 'BP33-8958', 
'BP58-12606', 'BP65-13108', 'BP37-10131', 'BP26-13929', 'BP02-7442', 'BP28-8369', 'BP64-12952', 'BP76-13789', 'BP07-7954', 'BP15-12460']

# bernby70
bern_train_video_list =['6869955', '7285088', '7368585', '8181232', '7353788', '7413013', '7440866', '6691243', '6963018', '7689789', '7539205', '7196916', '7321731', 
'6645369', '6959649', '7362800', '6690637', '6799808', '7539257', '7554678', '7599612', '7179089', '6880431', '6751003', '8190181', '6688650', '6978568', '7807683', '7689798',
'6751276', '5550090', '8116563', '7680366', '7406291', '7740041', '7353629', '6697612', '7511561', '7359903', '7000439']
bern_val_video_list = ['7127769', '7582326', '7021875', '6619966', '6810535', '6745206', '8294351', '7274073', '7498444', '6979132']
bern_test_video_list = ['6902252', '7556163', '6919206', '7382483', '8154586', '7668334', '6728101', '7196931', '6722627', '7169534', '5887959', '8341416', '7196928', '7350446', 
'7582149', '6629419', '6809750', '7262471', '7704520', '7285952']

stras_video_mode_dict = {
    'train': stras_train_video_list,
    'val': stras_val_video_list,
    'test': stras_test_video_list
}

bern_video_mode_dict = {
    'train': bern_train_video_list,
    'val': bern_val_video_list,
    'test': bern_test_video_list
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
        try:
            frame_data = label_data[frame_idx]
        except:
            print(f"Frame {frame_idx} not found in label data")
            continue
        event_label = np.where(frame_data == 1)[0][0]
        event_data.append(event_label)
    event = Counter(event_data).most_common(1)[0][0]
    current_event_label = deepcopy(event)

    # # display how often the transition is happening
    # cur_event = -1
    # for i in range(len(label_data)):
    #     frame_data = label_data[i]
    #     event_label = np.where(frame_data == 1)[0][0]
    #     if event_label != cur_event:
    #         print(f"Event transition from {cur_event} to {event_label} at frame {i}")
    #         cur_event = event_label
    # breakpoint()

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
        # set for next subsequent phase transitions NOTE: below computes for all. 
        # you need for the next not for all
        # for i in range(current_event_label+1, num_classes-1):
        #     try:
        #         time_to_next_event = next_event_time_in_min[i-current_event_label-1]
        #     except:
        #         breakpoint()
        #         #raise ValueError(f"Index out of bounds for the clip with start time {start_t} and end time {end_t}")
        #         # print(f"Index out of bounds for the clip with start time {start_t} and end time {end_t}")
        #         continue
        #     if time_to_next_event <= ne:
        #         event_onehot[i] = round(time_to_next_event, 4)
        #     else:
        #         break
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

def calculate_anticipation_labels(clip_metadata, 
                            frame_stride=60, 
                            clip_duration=4,
                            s_norm=10,
                            center='stras'):
    """
    Calculate the anticipation labels for the given clip metadata
    """
    modes = ['train', 'val', 'test']
    all_antr_data = {}
    for mode in modes:
        all_antr_data[mode] = {}
    
    for mode, mode_data in clip_metadata.items():
        video_antr_data = {}
        for video_name, video_data in mode_data.items():
            step_labels = video_data['step_label_matrix']
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
                
                # get rsd, NOTE: in RSDNet, for MBP140, the norm is 10 to keep RSD values within 0-20
                # TODO: check if they are actually within 0-20 range
                rsd = total_frames - clip_end_t - 1
                rsd = round(rsd / (60.0 * s_norm), 2)

                if rsd < 0 or rsd > 20:
                    # raise ValueError(f"RSD is out of range")
                    print(f"RSD is out of range for video {video_name}")
                    continue

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

def get_ann_meta(label_file):
    num_step_classes = 46 # 21 steps
    num_phase_classes = 12

    all_videos = {}
    video_labels = {}
    phase_classes = set()
    with open(label_file, 'rb') as f:
        ann_data = pickle.load(f)

    for mode, mode_ann in ann_data.items():
        video_labels[mode] = {}
        for video_name, video_ann in mode_ann.items():
            # total duration of the video in terms of extracted frames
            total_duration = len(video_ann)
            # collect the transition timestamps for phase and steps for each video
            for frame_ann in video_ann:
                frame_id = frame_ann['Frame_id']
                video_name = frame_id.split('_')[0]
                frame_int = int(frame_id.split('_')[-1])

                if video_name not in video_labels[mode]:
                    video_labels[mode][video_name] = {}
                    video_labels[mode][video_name]['gt'] = []
                
                ann_meta = {
                    'video_name': video_name,
                    'frame_id': frame_id,
                    'frame_int': frame_int,
                    'step_label': frame_ann['Step_gt'],
                    'frame_prefix': video_name + '_',
                    'phase_label': frame_ann['Phase_gt']
                }
                video_labels[mode][video_name]['gt'].append(ann_meta)
    
    for mode, mode_video_data in video_labels.items():
        all_videos[mode] = {}
        for video_name, video_data in mode_video_data.items():
            gt = video_data['gt']
            start_frame = int(gt[0]['frame_id'].split('_')[-1])
            end_frame = int(gt[-1]['frame_id'].split('_')[-1])
            video_labels_onehot = {'phase': [], 'step': []}
            for i, ann in enumerate(gt):
                frame_id = ann['frame_id']
                subfolder = 'MultiBypass140'
                frame_file = os.path.join('videos', subfolder, 'frames',video_name, frame_id+'.jpg')
                if not os.path.exists(frame_file):
                    print(f"Missing frame {frame_file} for video {video_name}")
                    continue
                
                frame_step_label = ann['step_label']
                frame_phase_label = ann['phase_label']
                step_one_hot = np.zeros(num_step_classes)
                phase_one_hot = np.zeros(num_phase_classes)

                try:
                    step_one_hot[int(frame_step_label)] = 1
                    phase_one_hot[int(frame_phase_label)] = 1
                except:
                    breakpoint()
                    print(f'Warning: Label {frame_step_label} or {frame_phase_label} not found in one hot encoding')

                video_labels_onehot['step'].append(step_one_hot)
                video_labels_onehot['phase'].append(phase_one_hot)

            video_labels_onehot['step'] = np.array(video_labels_onehot['step'])
            video_labels_onehot['phase'] = np.array(video_labels_onehot['phase'])
            all_videos[mode][video_name] = {}
            all_videos[mode][video_name]['step_label_matrix'] = video_labels_onehot['step']
            all_videos[mode][video_name]['phase_label_matrix'] = video_labels_onehot['phase']
            all_videos[mode][video_name]['frame_info'] = {'start_frame': start_frame, 'end_frame': end_frame}
    return all_videos

if __name__ == '__main__':
    path = "preprocessing/mbp140/label_files"
    center='bern'

    # label_files = os.path.join(path, f"{center}by70_1fps_fold0_total.pickle")
    label_files = os.path.join(path, "{center}by70_1fps_fold0_total.pickle")
    
    clip_duration = 8
    frame_stride = 20 # in seconds on frames that are extracted at 1 fps
    processed_labels = get_ann_meta(label_files)
    
    anticipation_labels = calculate_anticipation_labels(processed_labels,
                                            frame_stride=frame_stride, 
                                            clip_duration=clip_duration,
                                            s_norm=10,
                                            center=center)

    out_data = {}
    out_data['anticipation_labels'] = anticipation_labels
    out_data['clip_duration'] = clip_duration
    out_data['frame_stride'] = frame_stride
    
    # NOTE: uncomment below to save the anticipation labels to a json file
    # out_file = f'data/mbp140/label_files/frame{clip_duration}s_{center}_anticipation_v1.json'
    # with open(out_file, 'w') as f:
    #     json.dump(out_data, f, indent=2)