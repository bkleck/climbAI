import torch, torchvision
import argparse
import logging
import time
import numpy as np
import os
import json
import cv2
import random

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common functions I defined
import utils
from utils import *

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator

# import mediapipe for pose estimation
import mediapipe as mp

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


# create logging configs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logging.info(f'Is GPU being utilized?: {torch.cuda.is_available()}')

data_path = r"C:/Users/user\Desktop/RCP/ClimbAssistant/data" # BoonKong's
test_path = os.path.join(data_path, 'test')
output_dir = os.path.join(data_path, 'output')

# set cofigurations
configs = {'classes': 1}

# dictionary to store BGR values for each colour
colour_values = {"YELLOW": (0,255,255), "GREEN": (0,255,0), "BLUE": (255,0,0), "RED": (0,0,255), "BLACK": (0,0,0), "WHITE": (255,255,255)}

# settle the model configs
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = configs['classes']
predictor = DefaultPredictor(cfg)

# # get metadata from previous image runs
test_metadata = MetadataCatalog.get('val')



# read and write from webcam
start = time.time() 
cap = cv2.VideoCapture(0)

# define codec and get video details
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_name = os.path.join(test_path, 'webcam.mp4') # save output into a file called webcam.mp4
output_path = os.path.join(test_path, output_name)

# create video writer object for output of video file
writer = cv2.VideoWriter(filename=output_path, fourcc=fourcc,
                        fps=float(frames_per_second), 
                        frameSize=(width, height),
                        isColor=True,)

logging.info(f'Original FPS: {frames_per_second}')

# show if video could be opened or not
if (cap.isOpened() == False):
    logging.info(f'Error opening webcam!')
else:
    logging.info(f'Successfully opened webcam!')


count = 0 # use counter to ensure we do not run on every frame, too much time wasted (0.5s per inference)
startRun = False # use this Boolean to activate the recommender system
gotHolds = False # use this Boolean to check if we already have the contours of the holds

while (cap.isOpened()):
    ret, frame = cap.read()

    outputs = predictor(frame)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colour_list, contours = classifyHolds(outputs, hsv_frame)
    gotHolds = True
    logging.info("Obtained hold contours and colours from Mask-RCNN predictions.")

    output_img = frame.copy()

    # draw the contours we have obtained from our model with the correct colours
    for idx, contour in enumerate(contours):
        cv2.drawContours(output_img, [contour], -1, colour_values[colour_list[idx]], 3)
    
    cv2.imshow("window", output_img)
    cv2.waitKey(1)
