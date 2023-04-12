import torch, torchvision
import argparse
import logging
import time
import numpy as np
import os
import json
import cv2
import random

# for threading
import threading
import queue
import signal


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

# initialisation for audio output
import pyaudio

# Initlialise pyaudio stream & associated constants
p = pyaudio.PyAudio()
FS = 44100  # sampling rate, Hz, must be integer
CHANNELS = 2
FORMAT = pyaudio.paFloat32
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=FS,
                output=True)

# declare variables between thread
part = None
relative_dist = None
prev_part = False
prev_hold = False


# Define a function that will be run in a separate thread to call inference on every frame
def imageFunction(inference_output_q, stream):

    # while keyboard not interrupted
    while not stop_event.is_set():
        # Initialize mediapipe pose class.
        mp_pose = mp.solutions.pose
        # Setup the Pose function for images - independently for the images standalone processing.
        pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        data_path = os.path.join(os.getcwd(), 'data')
        test_path = os.path.join(data_path, 'src_test')
        output_dir = os.path.join(data_path, 'output_cosine')


        curr_time = time.time()
        log_file = os.path.join(test_path, f"log_{curr_time}.txt")
        # create logging configs
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        logging.info(f'Is GPU being utilized?: {torch.cuda.is_available()}')

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

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_file = 'webcam.mp4' + timestr
        output_name = os.path.join(test_path, output_file) # save output into a file called webcam.mp4
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


        inf_per_frame = 5 # used to call a prediction 1 in every n frame during start of run
        count = inf_per_frame # use counter to ensure we do not run on every frame, too much time wasted (0.5s per inference)
        startRun = False # use this Boolean to activate the recommender system
        gotHolds = False # use this Boolean to check if we already have the contours of the holds

        while (cap.isOpened()):
            ret, frame = cap.read()

            # if the system just started, we run Mask RCNN inference on first frame and save contours of holds
            if startRun == False:
                # if we have not gotten the holds, get them and change the variable gotHolds
                if gotHolds == False:
                    # get outputs from the Mask RCNN model
                    outputs = predictor(frame)
                    # create a HSV frame to do colour categorization
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    colour_list, contours_serializable = classifyHolds(outputs, hsv_frame)

                    contours = []
                    # deserialize the contours from json format into our numpy array
                    for contour_serializable in contours_serializable:
                        contour = []
                        for point_serializable in contour_serializable:
                            # convert each point to a numpy array with x and y coordinates and add it to contour
                            point = np.array([[point_serializable[0], point_serializable[1]]], dtype=np.int32)
                            contour.append(point)
                        # stack each contour horizontally to fit input format of openCV functions
                        contours.append(np.vstack([contour]))


                    gotHolds = True
                    logging.info("Obtained hold contours and colours from Mask-RCNN predictions.")

                # get the pose of human subject
                pose_results = detectPose(frame, pose_image)

                # only can check start condition if there is pose available
                if pose_results != None:
                    pose_img = frame.copy()
                    # get arm length for relative calculations
                    arm_length = getArmLength(pose_results)

                    # get hold touch condition for each body part
                    # threshold for hold to be considerec touching is distance = 10
                    threshold = 0.3
                    lw_condition = holdTouchCondition(contours, pose_results['l_wrist'], arm_length, threshold)
                    rw_condition = holdTouchCondition(contours, pose_results['r_wrist'], arm_length, threshold)
                    lf_condition = holdTouchCondition(contours, pose_results['l_foot'], arm_length, threshold)
                    rf_condition = holdTouchCondition(contours, pose_results['r_foot'], arm_length, threshold)
                    conditions = {'lw': lw_condition, 'rw': rw_condition, 'lf': lf_condition, 'rf': rf_condition}

                    text = []
                    text.append(f"left wrist: {lw_condition[1]}") 
                    text.append(f"right wrist: {rw_condition[1]}") 
                    text.append(f"left foot: {lf_condition[1]}") 
                    text.append(f"right foot: {rf_condition[1]}") 
                    
                    # draw lines from part to nearest holds
                    for k,v in conditions.items():
                        contour = v[0]
                        top = tuple(contour[contour[:, :, 1].argmin()][0])
                        
                        if k == 'lw':
                            cv2.line(pose_img, top, pose_results['l_wrist'], color=(255,255,255), thickness=2) 
                        elif k == 'rw':
                            cv2.line(pose_img, top, pose_results['r_wrist'], color=(255,255,255), thickness=2) 
                        elif k == 'lf':
                            cv2.line(pose_img, top, pose_results['l_foot'], color=(255,255,255), thickness=2) 
                        elif k == 'rf':
                            cv2.line(pose_img, top, pose_results['r_foot'], color=(255,255,255), thickness=2) 


                            # draw visualisations for pre-run
                            pose_img = drawOutputsBeforeRun(pose_img, contours, pose_results, colour_values, colour_list, text)

                            cv2.imshow("window", pose_img)
                            cv2.waitKey(1)
                            writer.write(pose_img)

                            start_result = checkStartCondition(colour_list, contours, lw_condition, rw_condition, lf_condition, rf_condition)

                            if start_result != False:
                                # keep only the same-coloured contours
                                colour, contours = start_result[0], start_result[1]

                                # get the contour of the highest hold - this is used to determine end-point
                                highest_hold = getLastHold(contours)

                                startRun = True # start the recommender system
                                logging.info("We have started our run.")
                                logging.info(f"User is climbing route with colour: {colour}")

                                output_img = frame.copy()
                                # draw the contours we have obtained from our model with the correct colours
                                for idx, contour in enumerate(contours):
                                    cv2.drawContours(output_img, [contour], -1, colour_values[colour], 3)

                                # draw circles on all essential body parts
                                for part in pose_results:
                                    cv2.circle(output_img, pose_results[part], 3, (0,255,0), 3)

                                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(test_path, "result.jpeg"), output_img[:, :, ::-1])

            else:
                if ret == True:
                    # vis_frame = frame # reset frame if nothing detected
                    # only run inference on 1 in every 30 frames
                    if count % inf_per_frame == 0:
                        print('frame detected')
                        start_1 = time.time()

                        pose_results = detectPose(frame, pose_image)

                        text = []

                        out = frame

                        # check if all pose keypoints are available
                        if pose_results != None:
                            print('pose detected')

                            # check if both wrists are touching the last hold
                            end_condition = checkEndCondition(highest_hold, pose_results, arm_length, threshold)

                            if end_condition == True:
                                logging.info("User has completed his run.")
                                quit()

                            # check to ensure only 1 limb off the holds (3-point contact)
                            # only then do we start our calculations for recommendation
                            part, on_hold = checkLimbs(contours, pose_results, arm_length, threshold)
                            # draw if no part off wall or nearest hold
                            out = drawOutputs(frame, contours, colour, colour_values, pose_results, text)

                            # In current iteration user has all limbs on hold but in previous iteration he has one limb off
                            # means the user just reached his next hold
                            # make a beep sound to let user know he has reached
                            if on_hold == 4 and prev_part != False and type(prev_hold) != bool:
                                # get the contour of the hold that the moving limb is currently touching
                                target_hold = holdTouchCondition(contours, pose_results[prev_part], arm_length, threshold)[0]

                                # if the hold he is currently touching is same as the hold he was supposed to touch
                                # release a short beep
                                if np.array_equal(target_hold, prev_hold):
                                    logging.info("User just touched a target hold.")
                                    beep(prev_part, stream)
                                    prev_part = False
                                    prev_hold = False


                            if part != False:
                                print('part detected')
                                prev_part = part # pass the part to the next iteration to check for just reach condition

                                logging.info(f'Part that is moving: {part}')
                                text.append(f'Part that is moving: {part}')
                                nearest_hold, text = nearestHold(contours, pose_results, part, arm_length, threshold, text)
                                out = drawOutputs(frame, contours, colour, colour_values, pose_results, text, nearest_hold)

                                # check if nearest holds are detected
                                if type(nearest_hold) != bool:
                                    print('nearest hold detected')
                                    prev_hold = nearest_hold

                                    relative_dist = calculateRelativeDistance(pose_results, nearest_hold, part, arm_length)
                                    text.append(f"distance: {relative_dist}")      
                                    stop_1 = time.time()
                                    diff_1 = float(stop_1 - start_1)
                                    # print(f'Time taken for predict step: {diff_1}')

                                    start_2 = time.time()
                                    out = drawOutputs(frame, contours, colour, colour_values, pose_results, text, nearest_hold)
                                    cv2.line(out, tuple(nearest_hold[nearest_hold[:, :, 1].argmax()][0]), pose_results[part], color=(255,255,255), thickness=2) 

                                    # Put inference output in the queue
                                    # we will pass the part and relative distance calculation to the sound thread
                                    queue_obj = (part, relative_dist)
                                    inference_output_q.put(queue_obj)
                                    
                                    stop_2 = time.time()
                                    diff_2 = float(stop_2 - start_2)
                                    # print(f'Time taken for draw step: {diff_2}')
                                
                                else:
                                    out = drawOutputs(frame, contours, colour, colour_values, pose_results, text, nearest_hold)

                    # output frame with predictions
                    vis_frame = out
                    writer.write(vis_frame)
                    cv2.imshow("window", vis_frame)
                    cv2.waitKey(1)

                    count += 1


                else:
                    break

logging.info(f'Completed inference on webcam.\n')




# Define a function that will be run in a separate thread to listen for the output
def soundFunction(inference_output_q, stream):
    """Thread to generate output sound concurrently with CV inference

    Args:
        inference_output_q (tuple): (part[str], relative_dist[float]). If not 3-point-contact, value is None.
    """
    
    while True:
        part, relative_dist = inference_output_q.get()
        output_bytes = write2stream(part=part, relative_dist=relative_dist, stream=stream)
        stream.write(output_bytes)



# Create a queue to pass inference output between the threads
inference_output_q = queue.Queue()

# Create a stop event to signal when the threads should stop
stop_event = threading.Event()

# Create and start the first thread to call inference on every frame
thread1 = threading.Thread(target=imageFunction, args=(inference_output_q,stream,))
thread1.start()

# Create and start the second thread to listen for the inference output
thread2 = threading.Thread(target=soundFunction, args=(inference_output_q,stream,))
thread2.start()

# Capture the KeyboardInterrupt signal
def signal_handler(signal, frame):
    print("Ctrl+C received, stopping threads...")
    stop_event.set()

# capture the CTRL + C signal on keyboard
signal.signal(signal.SIGINT, signal_handler)

# Wait for the threads to finish
thread1.join()
thread2.join()