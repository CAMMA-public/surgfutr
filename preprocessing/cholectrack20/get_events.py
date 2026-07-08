"""
gather the events (timestamps + count) from the CholecTrack20 dataset.

Events are :
1. occlusion
2. bleeding
3. undercoverage
4. reflection

"""

import os
import json
import glob
from copy import deepcopy

path = 'PATH TO CHOLECTRACK20 DATASET' # replace with the path to the cholectrack20 dataset
modes = ['Training', 'Validation', 'Testing']

events_dict = {
    'occluded': {'timestamps': [], 'onset':False, 'duration_between_events': []},
    'bleeding': {'timestamps': [], 'onset':False, 'duration_between_events': []},
    'smoke': {'timestamps': [], 'onset':False, 'duration_between_events': []}
}

all_splits_events = {
    'Training': {},
    'Validation': {},
    'Testing': {}
}

all_mode_video_lists = {
    'Training': [],
    'Validation': [],
    'Testing': []
}

for mode in modes:
    all_mode_video_lists[mode] = [os.path.basename(f) for f in glob.glob(os.path.join(path, mode, '*')) if not f.endswith('.tsv')]

# train, val, test - rdv split
train_video_list = ['VID01', 'VID15', 'VID26', 'VID40', 'VID52', 'VID65', 'VID79', 'VID02', 'VID18', 'VID27', 'VID43', 'VID56', 'VID66', 'VID92', 
                    'VID04', 'VID22', 'VID31', 'VID47', 'VID57', 'VID68', 'VID96', 'VID05', 'VID23', 'VID35', 'VID48', 'VID60', 'VID70', 'VID103', 
                    'VID13', 'VID25', 'VID36', 'VID49', 'VID62', 'VID75', 'VID110']
val_video_list = ['VID08', 'VID12', 'VID29', 'VID50', 'VID78']
test_video_list = ['VID06', 'VID51', 'VID10', 'VID73', 'VID14', 'VID74', 'VID32', 'VID80', 'VID42', 'VID111']

rdv_modes = {'Training': train_video_list, 'Validation': val_video_list, 'Testing': test_video_list}
# train videos not in rdv split:
train_unused_videos = [f for f in all_mode_video_lists['Training'] if f not in rdv_modes['Training'] and f not in rdv_modes['Validation'] and f not in rdv_modes['Testing']]
val_unused_videos = [f for f in all_mode_video_lists['Validation'] if f not in rdv_modes['Training'] and f not in rdv_modes['Validation'] and f not in rdv_modes['Testing']]
test_unused_videos = [f for f in all_mode_video_lists['Testing'] if f not in rdv_modes['Training'] and f not in rdv_modes['Validation'] and f not in rdv_modes['Testing']]

# remove VID11 from train_unused_videos and add to val_unused_videos
train_unused_videos.remove('VID11')
val_unused_videos.append('VID11')

for mode in rdv_modes:
    # adding folders following the rdv split
    folders = []
    for f in rdv_modes[mode]:
        # check if VID01 is in training, validation or testing
        for m in all_mode_video_lists:
            if f in all_mode_video_lists[m]:
                folders.append(os.path.join(path, m, f))

    if mode == 'Training':
        for f in train_unused_videos:
            folders.append(os.path.join(path, mode, f))
    elif mode == 'Validation':
        for f in val_unused_videos:
            if f != 'VID11':
                folders.append(os.path.join(path, mode, f))
            else:
                folders.append(os.path.join(path, 'Training', f))
    elif mode == 'Testing':
        for f in test_unused_videos:
            folders.append(os.path.join(path, mode, f))

    # for each mode initialize empty events_dict
    all_splits_events[mode] = {}

    for folder in folders:
        print(f'processing {folder}')
        videoname = os.path.basename(folder)
        all_splits_events[mode][videoname] = deepcopy(events_dict)
    
        # get the events.json file
        events_file = os.path.join(folder, f'{videoname}.json')
        all_splits_events[mode][videoname]['label_path'] = events_file
        with open(events_file, 'r') as f:
            events = json.load(f)
        events_annotations = events['annotations']

        # if mode == 'Testing':
        #     breakpoint()
        bl_start, oc_start, sm_start = -1, -1, -1

        for k, (frameid, labels) in enumerate(events_annotations.items()):
            frame_label = labels[0]

            # check for the cases where the bleeding is not 1 to get the timestamps where they occur
            if frame_label['bleeding'] == 1:
                if not all_splits_events[mode][videoname]['bleeding']['onset']: # bleeding is continuing
                    all_splits_events[mode][videoname]['bleeding']['timestamps'].append(frameid)
                    all_splits_events[mode][videoname]['bleeding']['onset'] = True
                    if bl_start == -1:
                        bl_start = frameid
            else: # save the duration of the bleeding event
                if all_splits_events[mode][videoname]['bleeding']['onset']:
                    dur = int(frameid) - int(bl_start) - 1
                    all_splits_events[mode][videoname]['bleeding']['duration_between_events'].append(dur)

                all_splits_events[mode][videoname]['bleeding']['onset'] = False
                bl_start = -1

            if frame_label['occluded'] == 1:
                if not all_splits_events[mode][videoname]['occluded']['onset']: # occlusion is continuing
                    all_splits_events[mode][videoname]['occluded']['timestamps'].append(frameid)
                    all_splits_events[mode][videoname]['occluded']['onset'] = True
                    if oc_start == -1:
                        oc_start = frameid
            else:
                if all_splits_events[mode][videoname]['occluded']['onset']:
                    dur = int(frameid) - int(oc_start) - 1
                    all_splits_events[mode][videoname]['occluded']['duration_between_events'].append(dur)
                all_splits_events[mode][videoname]['occluded']['onset'] = False
                oc_start = -1

            if frame_label['smoke'] == 1:
                if not all_splits_events[mode][videoname]['smoke']['onset']: # smoke is continuing
                    all_splits_events[mode][videoname]['smoke']['timestamps'].append(frameid)
                    all_splits_events[mode][videoname]['smoke']['onset'] = True
                    if sm_start == -1:
                        sm_start = frameid
            else:
                if all_splits_events[mode][videoname]['smoke']['onset']:
                    dur = int(frameid) - int(sm_start) - 1
                    all_splits_events[mode][videoname]['smoke']['duration_between_events'].append(dur)
                all_splits_events[mode][videoname]['smoke']['onset'] = False
                sm_start = -1

# display per mode in how many frames each event occurs
for mode in modes:
    print(f'{mode} split:')
    for videoname in all_splits_events[mode]:
        for event in events_dict:
            try:
                total_timestamps = len(all_splits_events[mode][videoname][event]['timestamps'])
            except:
                breakpoint()
            max_duration = max(all_splits_events[mode][videoname][event]['duration_between_events']) if all_splits_events[mode][videoname][event]['duration_between_events'] else -1
            min_duration = min(all_splits_events[mode][videoname][event]['duration_between_events']) if all_splits_events[mode][videoname][event]['duration_between_events'] else -1
            print(f"{videoname} : {event} occurs in {total_timestamps} timestamps! with max dur {max_duration} and min dur {min_duration}")

# get videos in train, val, test split
for mode in modes:
    print(f'{mode} split videos:')
    for videoname in all_splits_events[mode]:
        print(videoname)

# save all_splits_events to a json file
with open('data/cholectrack/all_splits_events_cholectrack20.json', 'w') as f:
    json.dump(all_splits_events, f)

# based on the dict and timestamps, check if the
breakpoint()
