import os
import json
import moviepy.editor as mp

# source video dataset - Cholec80
raw_video_path = '<YOUR_PATH>/cholec80/videos'

# to extract clips for state change from CholecT50 dataset
out_folder = '<YOUR_PATH>/ct50_state_change_clips'
label_file = 'data/state_change_analysis/clip3s_verb_state_change_v1.json'

# to extract clips for cystic structure anticipation from CholecT50 dataset, uncomment below
# out_folder = '<YOUR_PATH>/ct50_hct_triplet_clips'
# label_file = 'data/hct_triplets/clip3s_HCT_triplets_anticipation_v1.json'

with open(label_file, 'r') as f:
    clip_meta = json.load(f)

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

for split, video_list in clip_meta.items():
    for video_name, clip_list in video_list.items():
        video_int = int(video_name.split('VID')[-1])
        new_video_name = f'video{video_int:02d}'
        video_file = os.path.join(raw_video_path, f'{new_video_name}.mp4')

        # load video before extracting clips
        video = mp.VideoFileClip(video_file)
        assert video.fps == 25

        for clip in clip_list:
            # get current clip start and end
            cur_clip_name = clip['cur_clip_name']
            cur_clip_start = clip['start_t']
            cur_clip_end = clip['end_t']

            extract_clips(video,cur_clip_start, cur_clip_end, video_name, out_folder)

            if 'futr_clip_name' in clip:
                # get future clip start and end
                futr_clip_name = clip['futr_clip_name']
                futr_clip_start = futr_clip_name.split('_')[1].split('ps')[-1]
                futr_clip_end = futr_clip_name.split('_')[2].split('pe')[-1]

                extract_clips(video, futr_clip_start, futr_clip_end, video_name, out_folder)