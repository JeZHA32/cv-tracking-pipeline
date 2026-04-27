# Multi-Object Detection & Tracking Video Pipeline

End-to-end video processing pipeline built for the *Computer Vision* unit at La Trobe University (graded 75 / Distinction).

## What it does
1. Reads MP4 video input
2. Runs **Mask R-CNN** inference frame-by-frame on a GPU runtime
3. Filters detections per class with custom confidence thresholds
4. Assigns persistent track IDs frame-to-frame using the **Hungarian algorithm with IoU-based association**
5. Writes annotated MP4 output

## Stack
- **Python 3**, **PyTorch**, **OpenCV**, **NumPy**, **SciPy** (Hungarian)
- Pretrained **Mask R-CNN ResNet-50 FPN** from `torchvision`
- Designed to run on Google Colab with GPU acceleration

## Object classes detected
person · spoon · banana · apple · laptop · mouse · keyboard

## How tracking works
- Per-frame detections from Mask R-CNN
- Pairwise IoU matrix between current detections and active tracks
- Hungarian algorithm assigns optimal matches
- New tracks spawned for unmatched detections
- IDs persist across frames (and survive temporary occlusion)

## Notes
This was completed as a Master's coursework assignment. The video and detection cache files (`task1.mp4`, `all_boxes.pt`, `all_labels.pt`, `all_scores.pt`) are not included due to file size — the pipeline is adaptable to any MP4 input.
