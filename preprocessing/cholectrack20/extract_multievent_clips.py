"""
Given the timestamps of the events, extract the clips from the video
"""
import os
import json
import numpy as np
import moviepy.editor as mp

raw_video_path = 'PATH_TO_CHOLEC_RAW_VIDEOS' # replace with the path to the raw videos

def extract_clips(src_video, clip_start, clip_end, video_name, out_folder):
    subclip = src_video.subclip(clip_start, clip_end)

    assert subclip.fps == 25
    assert subclip.duration == 3

    # check if clip file already exists
    clip_name = f"{video_name}_ps{clip_start}_pe{clip_end}"
    out_path = os.path.join(out_folder, f"{clip_name}.mp4")
    if os.path.exists(out_path):
        return clip_name

    subclip.write_videofile(out_path, fps=25)

    return clip_name

def gen_clips_and_labels(all_splits_events, 
                         save_path, 
                         clip_duration=3,
                         save_clip=True, 
                         save_labels=True):
    """
    Extract the clips from the video
    """
    # extract the clips
    video_clips = {}
    clip_dist = {}
    for mode in all_splits_events:
        video_clips[mode] = {}
        clip_dist[mode] = {"bleeding": 0, "smoke": 0, "occluded": 0}
        for videoname in all_splits_events[mode]:
            # load video
            new_video_name = f'video{int(videoname.split("VID")[-1]):02d}'
            video_file = os.path.join(raw_video_path, f'{new_video_name}.mp4')
            video = mp.VideoFileClip(video_file)
            video_duration = video.duration

            for event in all_splits_events[mode][videoname]:
                if event not in ['bleeding', 'smoke', 'occluded']:
                    continue

                if event == 'bleeding':
                    key = 'timestamps'
                else:
                    key = 'filtered_timestamps'

                timestamps = all_splits_events[mode][videoname][event][key]
                timestamps = sorted(timestamps)

                # no bleeding events
                if len(timestamps) == 0:
                    continue

                # process valid event timestamps
                for frame_id in timestamps:# convert frame_id to seconds
                    frame_id = int(frame_id)
                    event_time = frame_id / 25

                    # extract clip times 1, 3, 5 seconds before the event.
                    for anticipation_duration in [1, 3, 5]:
                        start_t = int(max(0, event_time - anticipation_duration - clip_duration))
                        end_t = int(event_time - anticipation_duration)
                        # check if the end_t is smaller than video duration
                        if end_t > video_duration:
                            continue

                        if end_t - start_t < clip_duration:
                            print(f"Warning: Skipping clip at timestamp {frame_id} in {videoname} due duration {end_t - start_t} seconds")
                            continue

                        clip_id = f"{videoname}_ps{start_t}_pe{end_t}"
                        # save clip once the start and end timestamps are valid
                        if save_clip:
                            try:
                                clip_name = extract_clips(video, start_t, end_t, videoname, save_path)
                                assert clip_name == clip_id
                            except:
                                breakpoint()
                                continue

                        if clip_id not in video_clips[mode]:
                            video_clips[mode][clip_id] = {
                                "clip_id": clip_id,
                                "start_t": start_t,
                                "end_t": end_t,
                                "cur_clip_name": clip_id,
                                "event_t": frame_id,
                                "anticipation_duration": anticipation_duration,
                                "label": event}

                        clip_dist[mode][event] += 1

    # count the number of clips per mode train, val and test
    for mode in video_clips:
        print(f"Number of clips in {mode}: {len(video_clips[mode])}")
        print(f"Number of bleeding clips in {mode}: {clip_dist[mode]['bleeding']}")
        print(f"Number of smoke clips in {mode}: {clip_dist[mode]['smoke']}")
        print(f"Number of occluded clips in {mode}: {clip_dist[mode]['occluded']}")

    if save_labels:
        with open(os.path.join('data', 'cholectrack20_events_antr_v2.json'), 'w') as f:
            json.dump(video_clips, f)

if __name__ == '__main__':
    save_path = '/home/ssharma/datasets/surgical/cholectrack_clips'
    
    # load the events
    with open('data/cholectrack/all_splits_events_cholectrack20_filtered.json', 'r') as f:
        all_splits_events = json.load(f)
    
    gen_clips_and_labels(all_splits_events, 
                          save_path, 
                          clip_duration=3, 
                          save_clip=False, 
                          save_labels=False)