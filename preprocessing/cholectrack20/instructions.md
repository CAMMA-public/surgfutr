# CholecTrack20 Event Clips 🎬

This folder contains the preprocessing steps used to prepare CholecTrack20
video clips for bleeding, smoke, and occlusion event evaluation.

## Preparing the JSON File

1. 📥 Download the CholecTrack20 dataset from the official `CAMMA-public`
   GitHub repository.

2. 🩸 Run `get_events.py` to extract bleeding, smoke, and occlusion events from
   the dataset.
   Update the dataset path in the script before running it.

3. ✅ Run `check_events.py` to verify the extracted events and keep only the
   valid ones.
   This generates a filtered JSON file for downstream use.

4. ✂️ Run `extract_multievent_clips.py` to extract video clips for the valid
   events from the raw Cholec80 videos.
   Update the required paths in the script before running it.
