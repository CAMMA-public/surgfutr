
import json
import numpy as np

def find_stable_transitions(video_labels, 
                            src_label=0, 
                            dst_label=1, 
                            k=3, 
                            transition_list=[], 
                            filter_classes=[],
                            video_name=None):
    try:
        num_frames, num_classes = video_labels['label_matrix'].shape
    except:
        breakpoint()
    frame_info = video_labels['frame_info']
    label_matrix = video_labels['label_matrix']
    start_frame = frame_info['start_frame']
    end_frame = frame_info['end_frame']

    transitions = []
    existing_transitions = set(transition_list)
    num_skipped_transitions = 0
    for frame in range(start_frame, end_frame - k):
        # Skip if the current frame is within k frames of an existing transition
        if any(abs(frame - t) <= k for t in existing_transitions):
            continue

        frame_offset = frame - start_frame

        for cls in range(num_classes):
            if filter_classes and cls not in filter_classes:
                continue
            # Transition from src_label to dst_label
            try:
                cur_range = range(1, k + 1)
                futr_range = range(0, k)

                # add a check to see if the frame_offset are valid
                cur_valid_idx = [frame_offset - i for i in cur_range]
                futr_valid_idx = [frame_offset + i for i in futr_range]

                # NOTE: checks if the indices are valid (not less than 0 and not greater than num_frames)
                # if video_name != None:
                #     if video_name == 'pwhv8' and frame == 2767:
                #         breakpoint()

                # NOTE: as it is done per class, there might be some overlaps where the onsets occur within a span of few seconds.
                # THis is not a problem but something to keep in mind as its more pronounced when using frames instead of clips.
                if not all(i >= 0 and i < num_frames for i in cur_valid_idx) or not all(i >= 0 and i < num_frames for i in futr_valid_idx):
                    # print(f'Warning: Frame offset {frame_offset} is not valid for class {cls}')
                    continue

                if (label_matrix[frame_offset, cls] == dst_label and
                    all(label_matrix[frame_offset - i, cls] == src_label for i in cur_range) and
                    all(label_matrix[frame_offset + i, cls] == dst_label for i in futr_range)):
                    
                    # Check if this transition overlaps with existing transitions
                    if not any(abs(frame_offset - t) <= k for t in transitions):
                        transitions.append(frame)
                        break  # Move to next frame after finding a transition
                    else:
                        num_skipped_transitions += 1
            except:
                breakpoint()
    return transitions, num_skipped_transitions

def find_stable_transitions_with_label(video_verb_labels, src_label=0, dst_label=1, k=3, overlap_threshold=3, transition_list=[], filter_classes=[]):
    num_frames, num_classes = video_verb_labels.shape
    transitions = []
    existing_transitions = set(transition_list)

    num_multiple_cls = 0
    for frame in range(k, num_frames - k):
        # Skip if the current frame is within k frames of an existing transition
        if any(abs(frame - t) <= k for t in existing_transitions):
            continue
        
        flag = False
        selected_cls = []
        for cls in range(num_classes):
            if len(filter_classes) > 0 and cls not in filter_classes:
                continue
            # Transition from src_label to dst_label
            if (video_verb_labels[frame, cls] == dst_label and
                all(video_verb_labels[frame - i, cls] == src_label for i in range(1, k + 1)) and
                all(video_verb_labels[frame + i, cls] == dst_label for i in range(0, k))):
                
                # Check if this transition overlaps with existing transitions
                #if not any(abs(frame - t[0]) <= overlap_threshold for t in transitions):
                    # put the label as well excluding classes that are not in filter_classes
                    # transitions.append((frame, cls))
                #    selected_cls.append(cls) # NOTE: may be more than one class satisfies the transition
                #    flag = True

                selected_cls.append(cls) # NOTE: may be more than one class satisfies the transition
                flag = True

        if flag:
            transitions.append((frame, selected_cls))
            if len(selected_cls) > 1:
                num_multiple_cls += 1

    print(f'Number of multiple classes: {num_multiple_cls}')

    return transitions

def get_label_onehot(mode='verb', label_files=[]):
    if mode == 'verb':
        label_idx = 2
        num_classes = 10
    elif mode == 'triplet':
        label_idx = 0
        num_classes = 89
    elif mode == 'tool':
        label_idx = 1
        num_classes = 7

    all_videos = {}
    for e, label_file in enumerate(label_files):
        video_name = label_file.split('/')[-1].split('.')[0]
        with open(label_file, 'r') as f:
            data = json.load(f)
    
        annotations = data['annotations']
        num_ann = len(annotations)
        frame_keys = list(annotations.keys())
        frame_keys = sorted([int(i) for i in frame_keys])
        start_frame = frame_keys[0]
        end_frame = frame_keys[-1]

        # check if some frames are missing
        missing_frames = []
        for i in range(start_frame, end_frame + 1):
            if i not in frame_keys:
                missing_frames.append(str(i))

        video_labels = []
        count_dummy_labels = 0
        for i, ann_key in enumerate(range(start_frame, end_frame + 1)):
            ann_key = str(ann_key)
            if ann_key in missing_frames:
                frame_label = [-1.0]
                # print(f"Missing frame annotation at frame id {ann_key}!! Creating a dummy label..")
                count_dummy_labels += 1
            else:
                ann_data_list = annotations[ann_key]
                frame_label = []
                # the entries in ann_data could be one or more
                for ann in ann_data_list:
                    label = ann[label_idx]
                    frame_label.append(label)

            unique_labels = list(set(frame_label))
            label_one_hot = np.zeros(num_classes)
            for label in unique_labels:
                if label == -1.0:
                    # for missing frames, we will use -1.0 as the label
                    # continue
                    label_one_hot = [-1.0] * num_classes
                    break
    
                try:
                    label_one_hot[int(label)] = 1
                except:
                    breakpoint()
                    print(f'Warning: Label {label} not found in one hot encoding')
            
            video_labels.append(label_one_hot)
        
        video_labels = np.array(video_labels)
        all_videos[video_name] = {}
        all_videos[video_name]['label_matrix'] = video_labels
        if video_labels.shape[0] == 0:
            breakpoint()
            continue
        all_videos[video_name]['frame_info'] = {'start_frame': start_frame, 'end_frame': end_frame}
        assert video_labels.shape[0] == end_frame - start_frame + 1, f"Number of frames mismatch for VIDEO {video_name}: {video_labels.shape[0]} != {end_frame - start_frame + 1}"
        print(f"Number of dummy labels for VIDEO {video_name} with index {e}: {count_dummy_labels}")
        e += 1

    return all_videos

import os
def get_label_onehot_steps(label_files=[]):
    label_idx = 2
    num_classes = 21 # 21 steps
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

        # per_video_step_dist = {}
        # for video_name, video_data in video_labels.items():
        #     gt = video_data['gt']
        #     if video_name not in per_video_step_dist:
        #         per_video_step_dist[video_name] = {}
        #     print(f'Video {video_name} has {len(gt)} annotations')
        #     for ann in gt:
        #         step_id = ann['label']
        #         if step_id not in per_video_step_dist[video_name]:
        #             per_video_step_dist[video_name][step_id] = 0
        #         per_video_step_dist[video_name][step_id] += 1

        # print(f'Total number of steps: {len(per_video_step_dist)}')
        # for video_name, step_dist in per_video_step_dist.items():
        #     print(f'Video {video_name} has {len(step_dist)} steps and {sum(step_dist.values())} frames')
        #     # for step_id, count in step_dist.items():
        #     #     print(f'Step {step_id} has {count} frames')
        # breakpoint()

    # print(f'Total number of phase classes: {len(phase_classes)}')
    # breakpoint()

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
            label_one_hot = np.zeros(num_classes)
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
        try:
            assert video_labels_onehot.shape[0] == end_frame - start_frame + 1, f"Number of frames mismatch for VIDEO {video_name}: {video_labels_onehot.shape[0]} != {end_frame - start_frame + 1}"
        except:
            print(f"Number of frames mismatch for VIDEO {video_name}: {video_labels_onehot.shape[0]} != {end_frame - start_frame + 1}")
            continue
        e += 1
    return all_videos


def check_overlap(verb_onehot, 
                  onset_discontinuity_transitions, 
                  all_continuity_timestamps_1, 
                  all_continuity_timestamps_0):
    for video_name, video_verb_labels in verb_onehot.items():
        transitions = onset_discontinuity_transitions.get(video_name, [])
        continuity_timestamps_1 = all_continuity_timestamps_1.get(video_name, [])
        continuity_timestamps_0 = all_continuity_timestamps_0.get(video_name, [])

        # Extract frame numbers from continuity timestamps
        continuity_frames_1 = [c for c in continuity_timestamps_1]
        continuity_frames_0 = [c for c in continuity_timestamps_0]

        for t in transitions:
            if any(abs(t - c) <= 3 for c in continuity_frames_1) or any(abs(t - c) <= 3 for c in continuity_frames_0):
                print(f'Overlap found in {video_name} at transition {t}')
                breakpoint()

        # Optional: Check for overlaps between continuity timestamps
        for c1 in continuity_frames_1:
            if any(abs(c1 - c0) <= 3 for c0 in continuity_frames_0):
                print(f'Overlap found in {video_name} between continuity 1 ({c1}) and continuity 0')

        # Optional: Check for overlaps within continuity timestamps
        for i, c1 in enumerate(continuity_frames_1):
            if any(abs(c1 - c2) <= 3 for c2 in continuity_frames_1[i+1:]):
                print(f'Overlap found in {video_name} within continuity 1 timestamps')
                breakpoint()
                break

        for i, c0 in enumerate(continuity_frames_0):
            if any(abs(c0 - c2) <= 3 for c2 in continuity_frames_0[i+1:]):
                print(f'Overlap found in {video_name} within continuity 0 timestamps')
                break