# CholecT50 Clip Preparation

This folder contains scripts for preparing CholecT50 clips for state change
analysis and cystic structure anticipation.

## State Change Clips 🔄

1. 📝 Run `get_label_transitions.py` to generate a JSON file containing clips
   for onset and discontinuity events.
   No subsampling is applied at this stage.

2. ⏱️ To obtain continuity and background clips, apply subsampling later
   (for example, using a temporal window of 20 frames with frames extracted at
   1 FPS).
   The label stability duration is set to 3 seconds, matching the clip
   duration. Check the script for implementation details.

3. ✂️ Run `extract_video_clips.py` to extract the clips from the source
   videos using the JSON file generated in the previous step.
   The clips are saved to the specified output directory.

## Cystic Structure Anticipation Clips 🧠

1. 📝 Run `get_cystic_triplet_timestamps.py` to generate a JSON file for
   cystic structure anticipation.
   This step prepares the candidate tissue-deforming triplets and the clip
   timestamps needed for extraction.

2. 📦 The dataloader for this task handles the extracted clips and the
   anticipation duration used for training, validation, and testing.

3. ✂️ Run `extract_video_clips.py` to extract the clips from the source
   videos using the JSON file generated in the previous step.
   The clips are saved to the specified output directory.

## RSD and Phase Transition Labels 

1. Run `preprocessing/add_downstream_labels.py` to add RSD and phase transition labels to the JSON files generated in Step 1 under "State Change Clips" sections. This step enriches the dataset with additional labels for downstream tasks.