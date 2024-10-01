# Import necessary libraries
import cv2 as cv
import argparse
import numpy as np
import os
import sys
import random

# Initialize parameters
confThreshold = 0.5  # Confidence threshold for detection
maskThreshold = 0.3  # Mask threshold for segmentation

# Argument parser
parser = argparse.ArgumentParser(description='Mask-RCNN object detection and segmentation using OpenCV.')
parser.add_argument('--image', help='Path to the image file.')
parser.add_argument('--video', help='Path to the video file.')
args = parser.parse_args()

# Function to draw the bounding box and mask
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    """Draw bounding box and mask on the frame."""
    # Draw the bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # Prepare the label for display
    label = f'{classes[classId]}: {conf:.2f}' if classes else f'{conf:.2f}'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Apply the mask on the frame
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom+1, left:right+1][mask]

    # Randomize colors for different instances
    color = colors[random.randint(0, len(colors) - 1)]
    frame[top:bottom+1, left:right+1][mask] = (0.3 * color + 0.7 * roi).astype(np.uint8)

    # Draw contours of the mask
    mask = mask.astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8)

# Function to process the results from the network and apply masks and bounding boxes
def postprocess(boxes, masks):
    """Extract bounding boxes and masks for detected objects."""
    frameH, frameW = frame.shape[:2]
    numDetections = boxes.shape[2]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])

            # Get bounding box coordinates
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left, top = max(0, left), max(0, top)
            right, bottom = min(frameW - 1, right), min(frameH - 1, bottom)

            # Extract the mask for the object
            classMask = masks[i, classId]

            # Draw bounding box and mask
            drawBox(frame, classId, score, left, top, right, bottom, classMask)

# Load class names
classesFile = "mscoco_labels.names"
if not os.path.exists(classesFile):
    print(f"Error: {classesFile} not found!")
    sys.exit(1)

with open(classesFile, 'r') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load color scheme
colorsFile = "colors.txt"
if not os.path.exists(colorsFile):
    print(f"Error: {colorsFile} not found!")
    sys.exit(1)

with open(colorsFile, 'r') as f:
    colors = [np.array(color.split(), dtype=np.float32) for color in f.readlines()]

# Model files
textGraph = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
modelWeights = "mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

# Load the network
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Input handling (image or video)
if args.image:
    if not os.path.isfile(args.image):
        print(f"Error: Image file {args.image} doesn't exist.")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_mask_rcnn_output.jpg'
elif args.video:
    if not os.path.isfile(args.video):
        print(f"Error: Video file {args.video} doesn't exist.")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_mask_rcnn_output.avi'
else:
    cap = cv.VideoCapture(0)
    outputFile = "mask_rcnn_output.avi"

# Initialize video writer for saving output (if video input)
if args.video or not args.image:
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Process frames
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Processing completed.")
        print(f"Output file is stored as {outputFile}")
        break

    # Create 4D blob from frame
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)

    # Get output of the network
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Postprocess the detections
    postprocess(boxes, masks)

    # Display FPS and Inference Time
    t, _ = net.getPerfProfile()
    label = f'Inference time: {t * 1000.0 / cv.getTickFrequency():.0f} ms'
    cv.putText(frame, label, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show output
    cv.imshow('Mask-RCNN Object Detection', frame)

    # Save output if video
    if args.video:
        vid_writer.write(frame.astype(np.uint8))

# Cleanup
cap.release()
if args.video or not args.image:
    vid_writer.release()
cv.destroyAllWindows()
