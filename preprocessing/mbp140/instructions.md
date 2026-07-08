# MultiBypass140 Transfer on RSD, Phase, and Step Transition

Using the MultiBypass140 dataset, prepare annotation file for cross procedure transfer learning on RSD, Phase, and Step transition tasks.

## Cross-Procedure Transfer 🔄

1. 📝 Run `get_ann_from_mbp140.py` to generate a pickle file harmonizing annotations available in the public MultiBypass140 dataset into per center files.

2. ✂️ Run `get_anticipation_labels.py` to generate annotation files in JSON format for each center, containing the frame based clips for RSD, Phase, and Step transition tasks.