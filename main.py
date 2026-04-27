#!/usr/bin/env python
# coding: utf-8

# # CSE5CV Assignment 2
# 
# Use this notebook to complete the coding portion of the assignment. Once you have finished, you can submit your code by downloading the notebook as a .ipynb file (File > Download > Download .ipynb) and submitting that file on LMS.

# In[ ]:


# Type your student ID number as an integer here.
# e.g. STUDENT_ID = 22222222
STUDENT_ID =21686875


# **Before you continue**
# 
# Running a CNN on video data can be quite slow, so it is a good idea to make use of Google Colab's GPU acceleration to speed things up. To enable this, click Runtime > Change runtime type and select "GPU" as the Hardware accelerator.

# Run the cell below to import modules that you are likely to need when completing the assignment.

# In[ ]:


import copy
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf

from google.colab import drive


# The first thing that we need to do is connect Google Colab to your Google Drive. This allows you to:
# 
# * Upload a video file to your Google Drive and read it from this notebook.
# * Save detections from this notebook to your Google Drive for later access so you don't need to run a detection model on your video multiple times.
# * Write output files to your Google Drive from this notebook so that you can download them for submission.
# 
# Here is a step-by-step guide for you to follow:
# 
# 1. In a new tab, go to https://drive.google.com/ and create a top-level folder for this assignment named `CSE5CV_Assignment`.
# 2. Double-click on `CSE5CV_Assignment` to open that folder.
# 3. Upload your video file (e.g. `task1.mkv`) into that folder. You can do this by dragging and dropping or using the "New" button.
# 4. Run the code cell below to connect your Google Drive with this notebook. You will need to grant permissions for this to work (follow the prompts).

# In[ ]:


drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive')


# Run the cell below to preview the first frame from your video. If your video is called something other than `task1.mkv` (e.g. `task1.mp4`), you will need to edit the first line.

# In[ ]:


filename = './task1.mp4'

vid = cv2.VideoCapture(filename)
_, img = vid.read()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 1, 1)
ax.imshow(img)
fig.tight_layout()


# Now that you have confirmed that your video is accessible and being read correctly, we can use Mask R-CNN to detect bounding boxes, object class labels, and confidence scores for each frame of the video.
# 
# The cell below will take a while to run, but the results will be saved to your Google Drive. This means that you should only ever have to run the cell below once, even if you close the notebook and come back to continue work at a later date. However, if you decide to re-record your video you will need to run the cell again to generate new detections.

# In[ ]:


def preprocess_image(image):
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image

maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn.eval()
if torch.cuda.is_available():
    maskrcnn.cuda()

# Go to the start of the video
vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Record how long the video is (in frames)
vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# For each frame, read it, give it to maskrcnn and record the detections
all_boxes = []
all_labels = []
all_scores = []
for i in range(vid_length):
    _, img = vid.read()
    if img is None:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        input_image = preprocess_image(img)
        if torch.cuda.is_available():
            input_image = input_image.cuda()
        result = maskrcnn(input_image)[0]

    all_boxes.append(result['boxes'].detach().cpu().numpy())
    all_labels.append(result['labels'].detach().cpu().numpy())
    all_scores.append(result['scores'].detach().cpu().numpy())
    if i % 20 == 0:
        print(f'{i+1:0d}/{vid_length}')

torch.save(all_boxes, 'all_boxes.pt')
torch.save(all_labels, 'all_labels.pt')
torch.save(all_scores, 'all_scores.pt')


# Now if you go to the `CSE5CV_Assignment` folder on Google Drive you should see three new files where the detections have been stored: `all_boxes.pt`, `all_labels.pt`, and `all_scores.pt`.
# 
# Let's load these detections from Google Drive now:

# In[ ]:


all_boxes = torch.load('all_boxes.pt')
all_labels = torch.load('all_labels.pt')
all_scores = torch.load('all_scores.pt')

vid_length = len(all_boxes)

print(f'Loaded detections for {vid_length} video frames')


# Now it's your turn to complete the remaining tasks of the assignment by adding code in code cells below. You will find the "Pedestrian Tracking" lab coding notebook to be a useful reference for completing the assignment.
# 
# _You can create additional code cells by clicking the "+ Code" button in the toolbar._

# Part1:
# 
# Initial video output:
# Read specific frames(0,10,20,30) from the task1.mp4, I shot and draw bounding boxes on them for visualization. The detection results of each frame contain the class label and confidence score of the object, and are drawn on the image as rectangular boxes of different colors. Subsequently, all images containing detection boxes are displayed through matplotlib and saved as one image (detections.png).
# 
# Detection Filtering: Uses all_boxes, all_labels, and all_scores, which contains all raw detections from the model

# In[ ]:


# Define color palette for visualization
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]

def draw_detections(img, det, labels, scores, colours=COLOURS):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        i %= len(colours)
        label = labels[i]
        score = scores[i]
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=colours[i], thickness=2)
        label_score_str = str(label)+" "+str(score)
        cv2.putText(img, label_score_str, (tlx, tly+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[i], 2)
# Open the video file
vid = cv2.VideoCapture('./task1.mp4')

# Define frame numbers to process
frame_numbers = [0, 10, 20, 30]
n_rows = len(frame_numbers)

# Create figure for visualization
fig, axes = plt.subplots(n_rows, 1, figsize=(22, 13*n_rows))

for idx, fr_num in enumerate(frame_numbers):
    # Set video to specific frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, fr_num)

    # Read the frame
    _, img = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get detections for the current frame
    det = all_boxes[fr_num]
    labels = all_labels[fr_num]
    scores = all_scores[fr_num]

    # Convert detections to integer type
    det = det.astype(np.int32)

    # Draw detections on the image
    draw_detections(img, det, labels, scores)

    # Display the image
    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(f'Frame #{fr_num}')

# Save the figure
plt.savefig('./detections.png')

# Close the video file
vid.release()


#  Part2:
#  loading object detection results (including labels, bounding boxes, and scores) and assigning colors and names to each class for visualization purposes.

# In[ ]:


# Load detection results (labels, bounding boxes, and scores) from .pt files
all_labels = torch.load('all_labels.pt')  # List of object class labels for each frame
all_boxes = torch.load('all_boxes.pt')    # List of bounding boxes for each frame (x1, y1, x2, y2)
all_scores = torch.load('all_scores.pt')  # List of confidence scores for each detected object in each frame

# Print detection results for the first frame
print(all_labels[0])  # Print labels for the first frame
print(all_boxes[0])   # Print bounding boxes for the first frame
print(all_scores[0])  # Print confidence scores for the first frame

# Print the number of detected objects for the first frame
print(len(all_labels[0]), len(all_boxes[0]), len(all_scores[0]))  # Ensure that the number of labels, boxes, and scores match

# Class index mapping (these indices correspond to object categories)
# class_index==1 is person
# class_index==50 is spoon
# class_index==52 is banana
# class_index==53 is apple
# class_index==73 is laptop
# class_index==74 is mouse
# class_index==76 is keyboard
class_index = [1, 50, 52, 53, 73, 74, 76]  # These are the object class indices including in my video

# Assign a unique and easily distinguishable color (in BGR format) to each object class
# The colors are selected to help visually differentiate between object types
color_dict = {
    1: (0, 0, 255),     # Red - 'person'
    50: (255, 0, 0),    # Blue - 'spoon'
    52: (0, 255, 255),  # Yellow - 'banana'
    53: (0, 255, 0),    # Green - 'apple'
    73: (255, 0, 255),  # Purple - 'laptop'
    74: (255, 165, 0),  # Orange - 'mouse'
    76: (128, 0, 128)   # Dark purple - 'keyboard'
}

# Create a dictionary mapping class indices to class names
# This will be used for labeling detected objects during visualization
class_dict = {
    1: 'person',
    50: 'spoon',
    52: 'banana',
    53: 'apple',
    73: 'laptop',
    74: 'mouse',
    76: 'keyboard'
}


# Part3:
# Filter and process the results of object detection
# 
# Set a global confidence score threshold (e.g. 0.5) to filter out detections with lower confidence scores.
# Use a dictionary to specify a specific threshold for each class.

# In[ ]:


# Define categories and thresholds
class_index=[1,50,52,53,73,74,76]
# Set specific thresholds for each class
thresholds = {1: 0.9, 50: 0.6, 52: 0.7, 53: 0.45, 73: 0.6, 74: 0.6, 76: 0.6}

# Process each test result
filtered_labels = []
filtered_boxes = []
filtered_scores = []

for i in range(len(all_labels)):
    labels = all_labels[i]
    boxes = all_boxes[i]
    scores = all_scores[i]

    temp_labels = []
    temp_boxes = []
    temp_scores = []

    # Iterate over all labels
    for j in range(len(labels)):
        label = labels[j]
        score = scores[j]
        box = boxes[j]

        # Only keep the detection results whose confidence level is higher than the threshold of the category
        if label in class_index and score >= thresholds.get(label, 0.5):  # 默认全局阈值为 0.5
            temp_labels.append(label)
            temp_boxes.append(box)
            temp_scores.append(score)

    filtered_labels.append(temp_labels)
    temp_boxes = np.array(temp_boxes)
    filtered_boxes.append(temp_boxes)
    filtered_scores.append(temp_scores)

# Output the trimmed result
print(filtered_labels[30])
print(filtered_boxes[30])
print(filtered_scores[30])

# Output the length of each set after pruning
print(len(filtered_labels[30]), len(filtered_boxes[30]), len(filtered_scores[30]))


#  Part4:
#  loading object detection results (including labels, bounding boxes, and scores) and assigning colors and names to each class for visualization purposes.
# 
#  Processes and print more frames(300-500) (I appear in the latter part of the video)

# In[ ]:


COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']

]
def draw_detections(img, det, colours=COLOURS):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        i %= len(colours)
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=colours[i], thickness=2)

def draw_detections(img, det, labels, scores):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        label = labels[i]
        score = scores[i]
        cv2.rectangle(img, (tlx, tly), (brx, bry), color=color_dict[label], thickness=2)
        label_score_str = str(label)+" "+str(score)
        cv2.putText(img, label_score_str, (tlx, tly+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[label], 2)



frame_numbers = [0, 10, 20, 30, 300 ,350 ,400 ,450 , 500]
n_rows = len(frame_numbers)



#  open video with cv2.VideoCapture
vid = cv2.VideoCapture('./task1.mp4')

fig, axes = plt.subplots(n_rows, 1, figsize=(22, 13*n_rows))
for idx, fr_num in enumerate(frame_numbers):
    #  Set vid to frame i
    vid.set(cv2.CAP_PROP_POS_FRAMES, fr_num)

    #  Read a frame
    _, img = vid.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  Get detections for frame i out of all_detections
    det = filtered_boxes[fr_num]
    labels = filtered_labels[fr_num]
    scores = filtered_scores[fr_num]

    # Change det to the np.int32 dtype
    det = det.astype(np.int32)

    #  Draw on img
    draw_detections(img, det, labels, scores)

    axes[idx].imshow(img)
    axes[idx].axis('off')
    axes[idx].set_title(f'Frame #{fr_num}')
plt.savefig('./detections.png')


# Part5:
# The detections have been filtered based on specific criteria such as confidence threshold or object class.
# Each object class is assigned a specific color, making it easier to distinguish object types.
# The entire video is processed and a new video file is output.
# But this part is still missing apply object tracking to the detections using either IoU as the association method will be added in the final part.

# In[ ]:


vid = cv2.VideoCapture('./task1.mp4')
vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_out = cv2.VideoWriter('./task3.mp4', fourcc, 30, (720, 1280))

# fig, axes = plt.subplots(2, 1, figsize=(22, 13*2))
for fr_num in range(vid_length):
    # vid.set(cv2.CAP_PROP_POS_FRAMES, fr_num)
    # Read a frame
    _, img = vid.read()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    # break

    # TODO: Get detections for frame i out of all_detections
    det = filtered_boxes[fr_num]
    labels = filtered_labels[fr_num]
    scores = filtered_scores[fr_num]

    # TODO: Change det to the np.int32 dtype
    det = det.astype(np.int32)

    # TODO: Draw on img
    draw_detections(img, det, labels, scores)

    # axes[fr_num].imshow(img)
    # axes[fr_num].axis('off')
    # axes[fr_num].set_title(f'Frame #{fr_num}')
    # if fr_num>0:
    #     break

    vid_out.write(img)

vid_out.release()
from google.colab import files
files.download('./task3.mp4')


# Part 6:
# This code implements an IoU (intersection over union) based object tracking algorithm and overlays the tracking results on the original video
# 
# 1) The bbox_iou_matrix function is used to efficiently calculate the IoU between detection boxes.
# 
# 2) The track_objects function associates the detection results with existing tracks using the Hungarian algorithm and creates a new track for each unmatched detection.
# 
# 3)The draw_detections function draws the bounding box, label, confidence, and track ID on each frame.
# 
# 4) The main loop processes the video frame by frame, applies object tracking, and writes the results to an output video named "task3.mp4".
# The code is designed to ensure that each track contains only detection results of a single category.

# In[ ]:


from scipy.optimize import linear_sum_assignment

def bbox_iou_matrix(a, b):
    a = a[:, None]
    b = b[None, :]

    # Extract coordinates
    tlx_a, tly_a, brx_a, bry_a = [a[..., i] for i in range(4)]
    tlx_b, tly_b, brx_b, bry_b = [b[..., i] for i in range(4)]

    # Calculate intersection
    tlx_overlap = np.maximum(tlx_a, tlx_b)
    tly_overlap = np.maximum(tly_a, tly_b)
    brx_overlap = np.minimum(brx_a, brx_b)
    bry_overlap = np.minimum(bry_a, bry_b)

    intersection = (brx_overlap - tlx_overlap).clip(0) * (bry_overlap - tly_overlap).clip(0)

    # Calculate areas and union
    area_a = abs((brx_a - tlx_a) * (bry_a - tly_a))
    area_b = abs((brx_b - tlx_b) * (bry_b - tly_b))
    union = area_a + area_b - intersection

    return intersection / union

def draw_detections(img, det, labels, scores, track_ids):
    for box, label, score, track_id in zip(det, labels, scores, track_ids):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color_dict[label], thickness=2)
        label_text = f"{class_dict[label]} {score:.2f} ID:{track_id}"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color=color_dict[label], thickness=2)

def track_objects(detections, labels, scores, prev_tracks, iou_threshold=0.3):
    if not prev_tracks:
        return [{'box': det, 'label': label, 'score': score, 'id': i}
                for i, (det, label, score) in enumerate(zip(detections, labels, scores))]

    prev_boxes = np.array([track['box'] for track in prev_tracks])
    iou_matrix = bbox_iou_matrix(detections, prev_boxes)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T

    unmatched_detections = set(range(len(detections)))
    unmatched_tracks = set(range(len(prev_tracks)))

    new_tracks = []
    for d, t in matched_indices:
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.add(d)
            unmatched_tracks.add(t)
        else:
            new_tracks.append({
                'box': detections[d],
                'label': labels[d],
                'score': scores[d],
                'id': prev_tracks[t]['id']
            })
            unmatched_detections.remove(d)
            unmatched_tracks.remove(t)

    # Add new tracks for unmatched detections
    for d in unmatched_detections:
        new_tracks.append({
            'box': detections[d],
            'label': labels[d],
            'score': scores[d],
            'id': max([track['id'] for track in prev_tracks] + [t['id'] for t in new_tracks]) + 1
        })

    return new_tracks

# Main script
vid = cv2.VideoCapture('./task1.mp4')
vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_out = cv2.VideoWriter('./task3.mp4', fourcc, 30, (720, 1280))

tracks = []

for fr_num in range(vid_length):
    _, img = vid.read()

    det = filtered_boxes[fr_num]
    labels = filtered_labels[fr_num]
    scores = filtered_scores[fr_num]

    det = det.astype(np.int32)

    tracks = track_objects(det, labels, scores, tracks)

    track_ids = [track['id'] for track in tracks]
    draw_detections(img, det, labels, scores, track_ids)

    vid_out.write(img)

vid_out.release()

from google.colab import files
files.download('./task3.mp4')

