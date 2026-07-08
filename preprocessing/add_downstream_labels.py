# from the existing labels, add the downstream labels to the dataset - rsd and phase transition

import json
import os
import os.path as osp
from collections import Counter

video_dir = '<<YOUR_PATH>>/CholecT50/videos'
label_dir = '<<YOUR_PATH>>/CholecT50/labels'

ann_files = 'data/state_change/clip3s_verb_state_change_v1.json'
ann_data = json.load(open(ann_files, 'r'))

video_folders = {}
for f in os.listdir(video_dir):
    if osp.isdir(osp.join(video_dir, f)):
        video_folders[f] = len(os.listdir(osp.join(video_dir, f)))

def get_next_phase_transition(start_t, end_t, video_name, split='train'):
    # load the json file specific to the video
    label_file = osp.join(label_dir, f'{video_name}.json')
    label_data = json.load(open(label_file, 'r'))['annotations']
    num_frames = len(label_data)
    disable_clip = False

    # get the phase for the current clip majority vote
    phase_data = []
    for frame_idx in range(start_t, end_t+1): # end_t is inclusive so added 1 NOTE
        frame_data = label_data[str(frame_idx)]
        for instance in frame_data:
            phase_label = instance[-1]
            break
        phase_data.append(phase_label)

    phase = Counter(phase_data).most_common(1)[0][0]
    current_phase_label = phase

    # look for next phase transition (seconds and the class label)
    next_phase_time_in_sec = []
    num_next_phase_transitions = 0
    for frame_idx in range(end_t, num_frames):
        frame_data = label_data[str(frame_idx)]
        for i, instance in enumerate(frame_data):
            phase_label = instance[-1]
            if phase_label != phase:
                next_phase_tuple = (frame_idx - end_t, phase_label)
                next_phase_time_in_sec.append(next_phase_tuple)
                num_next_phase_transitions += 1
                phase = phase_label

        # if num_next_phase_transitions > 1: # clips from last phase will be disabled?
        #     break

    if num_next_phase_transitions == 0:
        disable_clip = True
        return [], disable_clip

    # now compute phase transition labels for the next 2, 3, 5 minutes
    time2nextphase = next_phase_time_in_sec[0][0] # in seconds
    if time2nextphase < 0:
        print(f"Time to next phase is negative for video {video_name} with start time {start_t} and end time {end_t}")
        disable_clip = True
        return [], disable_clip
    time2nextphase_min = time2nextphase / 60.0

    next_phase_label = next_phase_time_in_sec[0][1]
    
    # check if the 2nd is more than 5 mins NOTE: this is fine as it can occur in some videos
    # second_next_phase_label = next_phase_time_in_sec[1][1]
    # second_next_phase_time_in_min = next_phase_time_in_sec[1][0] / 60.0
    # if second_next_phase_time_in_min <= 5.0:
    #     print(next_phase_time_in_sec)
    #     breakpoint()
    #     raise ValueError(f"Second next phase is less than 5 mins for the clip {video_name} with start time {start_t} and end time {end_t}")
    
    phase_all_slots = []
    futr_time_slots = [2, 3, 5] # in min
    # for ne in futr_time_slots:
    #     phase_onehot = [ne] * 7 # total number of phases is 7
    #     phase_onehot[current_phase_label] = 0
    #     if time2nextphase_min <= ne:
    #         phase_onehot[next_phase_label] = round(time2nextphase_min, 4)
    #     phase_all_slots.extend(phase_onehot)

    # convert next_phase_time_in_sec to min NOTE: this does it for all phases as some phases occur immediately after the next phase
    next_phase_time_in_min = [x[0] / 60.0 for x in next_phase_time_in_sec]
    for ne in futr_time_slots:
        phase_onehot = [ne] * 7 # total number of phases is 7
        phase_onehot[current_phase_label] = 0
        # set for next subsequent phase transitions
        for i in range(current_phase_label+1, 6):
            try:
                time_to_next_phase = next_phase_time_in_min[i-current_phase_label-1]
            except:
                breakpoint()
                raise ValueError(f"Index out of bounds for the next phase time in min for the clip {video_name} with start time {start_t} and end time {end_t}")
            if time_to_next_phase <= ne:
                phase_onehot[i] = round(time_to_next_phase, 4)
            else:
                break
        phase_all_slots.extend(phase_onehot)

    # if video_name == 'VID01' and start_t == 1 and end_t == 4:
    #     breakpoint()

    return phase_all_slots, disable_clip
    # return next_phase_time_in_sec, disable_clip
def get_next_instrument_transition(start_t, end_t, video_name, split='train'):
    instrument_classes = ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator']
    instrument_class_idx_to_label = {i: instrument_classes[i] for i in range(len(instrument_classes))}
    instrument_label_to_class_idx = {v: k for k, v in instrument_class_idx_to_label.items()}
    instrument_label_idx_to_ignore = [0, 2] # as they are most occuring
    new_instrument_idx_to_label = {1:0, 3:1, 4:2, 5:3}

    # load the json file specific to the video
    label_file = osp.join(label_dir, f'{video_name}.json')
    label_data = json.load(open(label_file, 'r'))['annotations']
    num_frames = len(label_data)
    disable_clip = False

    # get the instrument for the current clip majority vote # NOTE: there are frames with no instrument and assigned -1
    instrument_data = []
    for frame_idx in range(start_t, end_t+1): # end_t is inclusive so added 1 NOTE
        frame_data = label_data[str(frame_idx)]
        for instance in frame_data:
            instrument_label = instance[1]
            break
        instrument_data.append(instrument_label)

    instrument = Counter(instrument_data).most_common(1)[0][0]
    current_instrument_label = instrument

    # ignore clips 

    # look for next instrument transition (seconds and the class label)
    next_instrument_time_in_sec = []
    num_next_instrument_transitions = 0
    instrument_appeared = {1:False, 3:False, 4:False, 5:False}
    for frame_idx in range(end_t, num_frames):
        frame_data = label_data[str(frame_idx)]
        for i, instance in enumerate(frame_data):
            instrument_label = instance[1]
            
            # ignore cases where there are no instruments
            if instrument_label == -1:
                continue

            if instrument_label != instrument and instrument_label not in instrument_label_idx_to_ignore:
                if instrument_appeared[instrument_label]:
                    # Instrument already appeared in the next transition
                    continue
                instrument_appeared[instrument_label] = True
                next_instrument_tuple = (frame_idx - end_t, instrument_label)
                next_instrument_time_in_sec.append(next_instrument_tuple)
                num_next_instrument_transitions += 1
                instrument = instrument_label

        # if num_next_instrument_transitions > 1: # clips from last instrument will be disabled?
        #     break
    if num_next_instrument_transitions == 0:
        disable_clip = True
        return [], disable_clip

    # now compute instrument transition labels for the next 2, 3, 5 minutes
    time2nextinstrument = next_instrument_time_in_sec[0][0] # in seconds
    if time2nextinstrument < 0:
        print(f"Time to next instrument is negative for video {video_name} with start time {start_t} and end time {end_t}")
        disable_clip = True
        return [], disable_clip
    time2nextinstrument_min = time2nextinstrument / 60.0
    next_instrument_label = next_instrument_time_in_sec[0][1]
    
    instrument_all_slots = []
    futr_time_slots = [2, 3, 5] # in min # NOTE: 5 min  has no tools for CholecT50
    # futr_time_slots = [1, 2, 3]

    # convert next_instrument_time_in_sec to min NOTE: this does it for all instruments as some instruments occur immediately after the next instrument
    next_instrument_time_in_min = [(x[0] / 60.0, x[1]) for x in next_instrument_time_in_sec]
    for ne in futr_time_slots:
        instrument_onehot = [ne] * 4 # total number of instruments is 6 but we ignore 0 and 2 leaving 4 classes (sorted : 1, 3, 4, 5)
        # try:
        #     if current_instrument_label not in [-1, 0, 2]:
        #         instrument_onehot[new_instrument_idx_to_label[current_instrument_label]] = 0
        # except:
        #     breakpoint()

        # NOTE: there could be clips where -1, 0, 2 are present in the current clip and next either of 1, 3, 4, 5 occurs. this will be valid
        if current_instrument_label not in [-1, 0, 2]:
            instrument_onehot[new_instrument_idx_to_label[current_instrument_label]] = 0
        else:
            pass

        # check for each instrument class
        for (time_to_next_instrument, instrument_label) in next_instrument_time_in_min:
            if time_to_next_instrument <= ne:
                instrument_onehot[new_instrument_idx_to_label[instrument_label]] = round(time_to_next_instrument, 4)
            else:
                break
        instrument_all_slots.extend(instrument_onehot)

    return instrument_all_slots, disable_clip
    # return next_instrument_time_in_sec, disable_clip

splits = ['train', 'val', 'test']
#splits = ['val']
for split in splits:
    videos_in_split = ann_data[split]
    total_samples = 0
    for video in videos_in_split:
        clip_list = ann_data[split][video]

        # check total number of frames in the video
        total_frames = video_folders[video]

        for clip in clip_list:
            clip_name = clip['clip_id']
            start_time = clip['start_t']
            end_time = clip['end_t']
            rsd = total_frames - end_time - 1
            rsd = round(rsd / (60.0 * 5.0), 2)
            clip['rsd'] = {}
            clip['rsd']['disable_clip'] = rsd < 0
            clip['rsd']['value'] = rsd

            # phase transition ..
            # step 2. compute the next phase transition
            phase_all_slots, disable_clip = get_next_phase_transition(start_time, end_time, video, split=split)
            clip['phase_transition'] = {}
            clip['phase_transition']['disable_clip'] = disable_clip
            clip['phase_transition']['phase_all_slots'] = phase_all_slots

            # instrument transition
            instrument_all_slots, disable_clip = get_next_instrument_transition(start_time, end_time, video, split=split)
            clip['instrument_transition'] = {}
            clip['instrument_transition']['disable_clip'] = disable_clip
            clip['instrument_transition']['instrument_all_slots'] = instrument_all_slots

            total_samples += 1
    print("total_samples", total_samples)

# save the ann_data
#json.dump(ann_data, open('data/state_change/clip3s_verb_state_change_rsd_phase_transition.json', 'w'))
json.dump(ann_data, open('data/state_change/clip3s_verb_state_change_rsd_phase_instrument_transition.json', 'w'))