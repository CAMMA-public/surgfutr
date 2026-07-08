
import json
import numpy as np

def find_stable_transitions(video_verb_labels, src_label=0, dst_label=1, k=3, transition_list=[], filter_classes=[]):
    num_frames, num_classes = video_verb_labels.shape
    transitions = []
    existing_transitions = set(transition_list)

    num_transitions_atleast_2cls = 0
    num_skipped_transitions = 0
    for frame in range(k, num_frames - k):
        # Skip if the current frame is within k frames of an existing transition
        if any(abs(frame - t) <= k for t in existing_transitions):
            continue
        
        num_onset_transition_in_1frame = 0
        for cls in range(num_classes):
            if filter_classes and cls not in filter_classes:
                continue
            # Transition from src_label to dst_label
            if (video_verb_labels[frame, cls] == dst_label and
                all(video_verb_labels[frame - i, cls] == src_label for i in range(1, k + 1)) and
                all(video_verb_labels[frame + i, cls] == dst_label for i in range(0, k))):
                
                # Check if this transition overlaps with existing transitions
                if not any(abs(frame - t) <= k for t in transitions):
                    transitions.append(frame)
                    num_onset_transition_in_1frame += 1
                    break  # Move to next frame after finding a transition
                else:
                    num_skipped_transitions += 1
        
        if num_onset_transition_in_1frame >= 2:
            num_transitions_atleast_2cls += 1

        # NOTE: below considers all frames of a video
        # if num_onset_transition_in_1frame == 0:
        #     num_skipped_transitions += 1

    if num_transitions_atleast_2cls > 0:
        print(f'Number of transitions with at least 2 classes: {num_transitions_atleast_2cls}')

    # if num_skipped_transitions > 0:
    #     print(f'Number of skipped transitions: {num_skipped_transitions}')
    
    # return transitions # default without returning number of skipped transitions
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
        label_idx = 7
        num_classes = 10
    elif mode == 'triplet':
        label_idx = 0
        num_classes = 100
    elif mode == 'tool':
        label_idx = 1
        num_classes = 6
    elif mode == 'phase':
        label_idx = 14
        num_classes = 7

    all_videos = {}
    for label_file in label_files:
        video_name = label_file.split('/')[-1].split('.')[0]
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        num_ann = len(annotations)

        video_labels = []
        
        # collect verb labels for all the frames in the video to check for transitions later
        for i in range(num_ann):
            ann_key = str(i)
            ann_data_list = annotations[ann_key]

            frame_label = []

            # the entries in ann_data could be one or more
            for ann in ann_data_list:
                label = ann[label_idx]
                frame_label.append(label)

            # unique verb labels
            unique_labels = list(set(frame_label))

            # one hot encoding
            label_one_hot = np.zeros(num_classes)
            for label in unique_labels:
                if label == -1.0:
                    continue
                try:
                    label_one_hot[int(label)] = 1
                except:
                    breakpoint()
                    print(f'Warning: Label {label} not found in one hot encoding')
            
            video_labels.append(label_one_hot)

        video_labels = np.array(video_labels)
        all_videos[video_name] = video_labels

    return all_videos


def check_overlap(verb_onehot, 
                  onset_discontinuity_transitions, 
                  all_continuity_timestamps_1, 
                  all_continuity_timestamps_0,
                  stability_duration=3):
    for video_name, video_verb_labels in verb_onehot.items():
        transitions = onset_discontinuity_transitions.get(video_name, [])
        continuity_timestamps_1 = all_continuity_timestamps_1.get(video_name, [])
        continuity_timestamps_0 = all_continuity_timestamps_0.get(video_name, [])

        # Extract frame numbers from continuity timestamps
        continuity_frames_1 = [c for c in continuity_timestamps_1]
        continuity_frames_0 = [c for c in continuity_timestamps_0]

        for t in transitions:
            if any(abs(t - c) <= stability_duration for c in continuity_frames_1) or any(abs(t - c) <= stability_duration for c in continuity_frames_0):
                print(f'Overlap found in {video_name} at transition {t}')
                #breakpoint()

        # Optional: Check for overlaps between continuity timestamps
        for c1 in continuity_frames_1:
            if any(abs(c1 - c0) <= stability_duration for c0 in continuity_frames_0):
                print(f'Overlap found in {video_name} between continuity 1 ({c1}) and continuity 0')

        # Optional: Check for overlaps within continuity timestamps
        for i, c1 in enumerate(continuity_frames_1):
            if any(abs(c1 - c2) <= stability_duration for c2 in continuity_frames_1[i+1:]):
                print(f'Overlap found in {video_name} within continuity 1 timestamps')
                #breakpoint()
                #break

        for i, c0 in enumerate(continuity_frames_0):
            if any(abs(c0 - c2) <= stability_duration for c2 in continuity_frames_0[i+1:]):
                print(f'Overlap found in {video_name} within continuity 0 timestamps')
                #break