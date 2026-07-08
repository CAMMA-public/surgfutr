import json
import os
import pickle

path = "<YOUR-PATH>/MultiBypass140"
save_path = "preprocessing/mbp140/label_files"
modes = ["train", "val", "test"]

# NOTE: Comment/Uncomment below to use bernby70 or strasby70 dataset

# train_ann_file = os.path.join(path, "bernby70/labels/train", "1fps_100_0.pickle")
# val_ann_file = os.path.join(path, "bernby70/labels/val", "1fps_0.pickle")
# test_ann_file = os.path.join(path, "bernby70/labels/test", "1fps_0.pickle")

train_ann_file = os.path.join(path, "strasby70/labels/train", "1fps_100_0.pickle")
val_ann_file = os.path.join(path, "strasby70/labels/val", "1fps_0.pickle")
test_ann_file = os.path.join(path, "strasby70/labels/test", "1fps_0.pickle")

train_ann = pickle.load(open(train_ann_file, "rb"))
val_ann = pickle.load(open(val_ann_file, "rb"))
test_ann = pickle.load(open(test_ann_file, "rb"))

modes = ["train", "val", "test"]
new_video_ann = {}
for mode, mode_ann in zip(modes, [train_ann, val_ann, test_ann]):
    new_video_ann[mode] = {}
    video_names = list(mode_ann.keys())
    # print(f"Processing {mode} videos: {video_names}")
    for video_name, video_ann in mode_ann.items():
        for frame_ann in video_ann:
            frame_id = frame_ann['Frame_id']
            frame_path = os.path.join("frames", video_name, f"{frame_id}.jpg")
            frame_ann['Frame_path'] = frame_path
        full_video_name = frame_ann['Frame_id'].split('_')[0]
        if full_video_name not in new_video_ann[mode]:
            new_video_ann[mode][full_video_name] = []
        new_video_ann[mode][full_video_name].append(frame_ann)

# get list of all videos
print(f"list of train videos: {list(new_video_ann['train'].keys())}")
print(f"list of val videos: {list(new_video_ann['val'].keys())}")
print(f"list of test videos: {list(new_video_ann['test'].keys())}")

# NOTE: Uncomment below to collect all video anns into train, val and test
# combine them to a single pickle file with modes as keys
# combined_ann = {}
# for mode in modes:
#     combined_ann[mode] = eval(f"{mode}_ann")
# # pickle.dump(combined_ann, open(os.path.join(save_path, "bernby70_1fps_fold0_total.pickle"), "wb"))
# pickle.dump(combined_ann, open(os.path.join(save_path, "strasby70_1fps_fold0_total.pickle"), "wb"))
