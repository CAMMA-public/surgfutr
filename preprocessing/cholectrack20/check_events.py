import json
import os

def create_event_timestamps(all_splits_events):
    gap_threshold = 5 * 25 + 15 * 25 # 5 seconds with 25 fps + 15 seconds with 25 fps
    smoke_gap_threshold = 30 * 25 # 10 seconds with 25 fps
    occlusion_gap_threshold = 40 * 25 # 10 seconds with 25 fps
    modes = all_splits_events.keys()
    for mode in modes:
        total_events = 0
        total_bleeding_events = 0
        total_smoke_events = 0
        total_occlusion_events = 0

        for videoname in all_splits_events[mode]:
            total_selected_events = 0

            # get the bleeding timestamps first (all selected)
            bleeding_timestamps = sorted(all_splits_events[mode][videoname]['bleeding']['timestamps'])
            total_selected_events += len(bleeding_timestamps)
            total_bleeding_events += len(bleeding_timestamps)

            # get the smoke timestamps ..
            # 1. filter within smoke timestamps such that they are not within the gap threshold of
            # 2. check for the overlap with bleeding timestamps
            # 3. check if the smoke timestamps fall within the gap threshold of the bleeding timestamps

            smoke_timestamps = sorted(all_splits_events[mode][videoname]['smoke']['timestamps'])

            # 1. filter within smoke timestamps such that they are not within the gap threshold of each other
            smoke_frames = [int(ts) for ts in smoke_timestamps]
            filtered = [smoke_frames[0]]  # Always keep first timestamp
            filtered.extend(frame for i, frame in enumerate(smoke_frames[1:], 1) if frame - smoke_frames[i-1] >= smoke_gap_threshold)
            smoke_timestamps = [str(frame) for frame in sorted(filtered)]

            # 2. check for the overlap with bleeding timestamps
            filtered_smoke_timestamps = []
            for smoke_timestamp in smoke_timestamps:
                if smoke_timestamp in bleeding_timestamps:
                    continue

                # Check distance to nearest bleeding frame
                if len(bleeding_timestamps) > 0:
                    min_distance = min(abs(int(smoke_timestamp) - int(bleeding_timestamp)) 
                                    for bleeding_timestamp in bleeding_timestamps)
                else:
                    min_distance = smoke_gap_threshold

                # Add smoke frame if not within gap threshold
                if min_distance >= smoke_gap_threshold:
                    filtered_smoke_timestamps.append(str(smoke_timestamp))

            all_splits_events[mode][videoname]['smoke']['filtered_timestamps'] = filtered_smoke_timestamps
            total_selected_events += len(filtered_smoke_timestamps)
            total_smoke_events += len(filtered_smoke_timestamps)

            # get occlusion timestamps ..
            # 1. filter within occlusion timestamps such that they are not within the gap threshold of each other
            # 2. check for overlap with bleeding timestamps
            # 3. check for overlap with smoke timestamps
            # 4. check for gap threshold with bleeding timestamps

            occlusion_timestamps = sorted(all_splits_events[mode][videoname]['occluded']['timestamps'])

            occlusion_frames = [int(ts) for ts in occlusion_timestamps]
            filtered = [occlusion_frames[0]]  # Always keep first timestamp
            filtered.extend(frame for i, frame in enumerate(occlusion_frames[1:], 1) if frame - occlusion_frames[i-1] >= occlusion_gap_threshold)
            occlusion_timestamps = [str(frame) for frame in sorted(filtered)]

            filtered_occlusion_timestamps = []
            for occlusion_timestamp in occlusion_timestamps:
                if occlusion_timestamp in bleeding_timestamps or occlusion_timestamp in smoke_timestamps:
                    continue    

                if len(bleeding_timestamps) > 0:
                    min_distance = min(abs(int(occlusion_timestamp) - int(bleeding_timestamp)) 
                                    for bleeding_timestamp in bleeding_timestamps)
                else:
                    min_distance = occlusion_gap_threshold

                if min_distance >= occlusion_gap_threshold:
                    filtered_occlusion_timestamps.append(str(occlusion_timestamp))

            all_splits_events[mode][videoname]['occluded']['filtered_timestamps'] = filtered_occlusion_timestamps
            total_selected_events += len(filtered_occlusion_timestamps)
            total_occlusion_events += len(filtered_occlusion_timestamps)

            print(f"Split: {mode}, Video: {videoname}, Total selected events: {total_selected_events}")
            total_events += total_selected_events
        print(f"Total events: {total_events}")
        print(f"Total bleeding events >>>> : {total_bleeding_events}")
        print(f"Total smoke events >>>> : {total_smoke_events}")
        print(f"Total occlusion events >>>> : {total_occlusion_events}")

    return all_splits_events

if __name__ == '__main__':  

    event_file = 'data/cholectrack/all_splits_events_cholectrack20.json'
    with open(event_file, 'r') as f:
        all_splits_events = json.load(f)

    filtered_events = create_event_timestamps(all_splits_events)

    with open('data/cholectrack/all_splits_events_cholectrack20_filtered.json', 'w') as f:
        json.dump(filtered_events, f)