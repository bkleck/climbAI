import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import logging
import os
import requests
import datetime
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# create logging configs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Function that classify colour of each hold
# takes in input of Mask-RCNN predictions and the original RGB frame in HSV values
# returns a list of colours and contours for each hold
def classifyHolds(outputs, hsv_frame):   
    colour_list = []
    contours = []

    for pred_mask in outputs['instances'].pred_masks:
        mask = pred_mask.cpu().numpy().astype('uint8') # extract mask from predictions

        # Extract the contour of each predicted mask and save it in a list
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.append(contour[0])

        hsv_mean = cv2.mean(hsv_frame, mask=mask) # calculate mean HSV values of each mask

        # get hue and value
        # hue is between 0 and 180, value is between 0 and 255
        hue = hsv_mean[0]
        saturation = hsv_mean[1]
        value = hsv_mean[2]

        # make use of hue to categorize colours
        color = "Undefined"
        if hue < 40:
            color = "YELLOW"
        elif hue < 90:
            color = "GREEN"
        elif hue < 110:
            color = "BLUE"
        else:
            color = "RED"
        
        # however, value is used to determine black and white, and hue is not useful
        # hence we reset the color if any of the holds are too dark or light as they should be classified as black and white respectively
        if value < 80:
            color = "BLACK"
        elif saturation < 110 and value > 121:
            color = "WHITE" 

        colour_list.append(color)

    # serialize the contour arrays in such a way that it is able to be jsonified to be sent over AWS Lambda
    contours_serializable = []
    for contour in contours:
        contour_serializable = []
        for point in contour:
            contour_serializable.append(point[0].tolist())
        contours_serializable.append(contour_serializable)

    return colour_list, contours_serializable


# Test colour on actual wall
# Output a image with HSV value of each hold for checking
def classifyHoldsTest(outputs, hsv_frame, colour_values):   
    colour_list = []
    contours = []

    output_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

    for pred_mask in outputs['instances'].pred_masks:
        mask = pred_mask.cpu().numpy().astype('uint8') # extract mask from predictions

        # Extract the contour of each predicted mask and save it in a list
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # get top of contour
        top = tuple(contour[0][contour[0][:, :, 1].argmin()][0])
        contours.append(contour[0])

        hsv_mean = cv2.mean(hsv_frame, mask=mask) # calculate mean HSV values of each mask
        hsv_mean = tuple([int(x) for x in hsv_mean])
    
        # get hue and value
        # hue is between 0 and 180, value is between 0 and 255
        hue = hsv_mean[0]
        saturation = hsv_mean[1]
        value = hsv_mean[2]

        # make use of hue to categorize colours
        color = "Undefined"
        if hue < 48:
            color = "YELLOW"
        elif hue < 90:
            color = "GREEN"
        elif hue < 120:
            color = "BLUE"
        else:
            color = "RED"
        
        # however, value is used to determine black and white, and hue is not useful
        # hence we reset the color if any of the holds are too dark or light as they should be classified as black and white respectively
        if value < 48:
            color = "BLACK"
        elif saturation < 110 and value > 121:
            color = "WHITE" 

        colour_list.append(color)
        cv2.putText(output_frame, str(hsv_mean), org=top, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.35, color=colour_values[color], 
        thickness=1)


    cv2.imwrite(os.path.join("C:/Users/user/Desktop/RCP/ClimbAssistant/data/colour_test", "color.jpeg"), output_frame)

    # serialize the contour arrays in such a way that it is able to be jsonified to be sent over AWS Lambda
    contours_serializable = []
    for contour in contours:
        contour_serializable = []
        for point in contour:
            contour_serializable.append(point[0].tolist())
        contours_serializable.append(contour_serializable)

    return colour_list, contours_serializable


# Function to detect whether a human is in the frame or not
# Make use of Mask RCNN model in Detectron2 trained on COCO dataset
def detectHuman(image):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set high threshold to reduce false positives
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)


    # only get the classes if it is equal to 0 -- meaning it is a human
    classes = outputs["instances"][outputs["instances"].pred_classes == 0].pred_classes.tolist()

    if len(classes) > 0:
        return True
    else:
        return False


# Function to detect pose of person in image and extract keypoints
# In our use case, we are interested in the wrists, feet and hips coordinates
def detectPose(image, pose):
    
    image_in_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # openCV uses BGR, hence we need to convert to normal RGB
    
    resultant = pose.process(image_in_RGB) # perform pose detection

    keypoints = []

    # extract landmark coordinates from the defined variable and put them into a list
    # list has length of 33, for each body key-point detected
    if resultant.pose_landmarks:
        for data_point in resultant.pose_landmarks.landmark:
            keypoints.append({
                                'X': data_point.x,
                                'Y': data_point.y,
                                'Z': data_point.z
                                })
    
    # normalize the pixel values from mediapipe to draw using opencv
    image_rows, image_cols, _ = image.shape

    # make sure all the required keypoints are available
    if len(keypoints) == 33:
        left_wrist = _normalized_to_pixel_coordinates(keypoints[15]['X'],keypoints[15]['Y'],image_cols,image_rows)
        right_wrist = _normalized_to_pixel_coordinates(keypoints[16]['X'],keypoints[16]['Y'],image_cols,image_rows)
        left_elbow = _normalized_to_pixel_coordinates(keypoints[13]['X'],keypoints[13]['Y'],image_cols,image_rows)
        right_elbow = _normalized_to_pixel_coordinates(keypoints[14]['X'],keypoints[14]['Y'],image_cols,image_rows)
        # left_hip = _normalized_to_pixel_coordinates(keypoints[23]['X'],keypoints[23]['Y'],image_cols,image_rows)
        # right_hip = _normalized_to_pixel_coordinates(keypoints[24]['X'],keypoints[24]['Y'],image_cols,image_rows)
        left_foot = _normalized_to_pixel_coordinates(keypoints[31]['X'],keypoints[31]['Y'],image_cols,image_rows)
        right_foot = _normalized_to_pixel_coordinates(keypoints[32]['X'],keypoints[32]['Y'],image_cols,image_rows)

        results = {'l_wrist': left_wrist, 'r_wrist': right_wrist, 'l_elbow': left_elbow, 'r_elbow': right_elbow,
            'l_foot': left_foot, 'r_foot': right_foot}

        # check if we have None values in our dictionary
        none_values = not all(results.values())
        # if we do, do not return results
        if none_values:
            results = None
        else:
            results = results

    else:
        logging.error("Unable to detect required pose key-points of user.")
        results = None # do this to trigger the error
    
    return results



# Function to check if a body part is touching a hold
# Threshold value is used to determine how close the part must be to the hold to be considered touching
# Returns coordinates of hold and condition
def holdTouchCondition(contours, part, arm_length, threshold):
    # first check if the body part is inside or touching any contour
    # we get a list of contours that the body part is inside
    touched_hold = [contour for contour in contours if cv2.pointPolygonTest(contour, part, False) != -1]

    # if there is a contour that the body part is inside, return it
    if len(touched_hold) > 0:
        nearest_hold = touched_hold[0]
        condition = "touched"
        return [nearest_hold, condition]


    # if there are no contours the body part is inside, we need to check if it is near anywhere
    # within a certain threshold, we take it as the part is touching the hold
    else:
        holds = []
        for contour in contours:
            # calculate the nearest distance from the part to any point on the edge of the contour
            dist = cv2.pointPolygonTest(contour,part,True)

            # save to a list of dictionaries
            holds.append({'contour': contour, 'distance': abs(dist)})

        # sort by distance and get nearest contour and distance
        sorted_holds = sorted(holds, key=lambda i: i['distance'])

        nearest_hold = sorted_holds[0]['contour']
        nearest_dist = sorted_holds[0]['distance']

        # use arm length to get relative distance
        nearest_dist = round((nearest_dist / arm_length), 2)

        if nearest_dist < threshold:
            condition = "touched"
        else:
            condition = nearest_dist

        return [nearest_hold, condition]




# Function to check if user is in starting position
# If both wrists and feet of user are holding the same coloured holds, we can start our route recommender
def checkStartCondition(colours, contours, lw_condition, rw_condition, lf_condition, rf_condition):

    # if any part is not touching any contour, we cannot start our system
    if (lw_condition[1] != "touched") or (rw_condition[1] != "touched") or (lf_condition[1] != "touched") or (rf_condition[1] != "touched"):
        logging.info("User does not have both hands and feet on the holds yet.")
        return False
    
    else:
        # get indexes of the holds to check their colours
        try:
            lw_idx = contours.index(lw_condition[0])
            rw_idx = contours.index(rw_condition[0])
            lf_idx = contours.index(lf_condition[0])
            rf_idx = contours.index(rf_condition[0])

            # if both hold colours are the same, we can start the system
            if colours[lw_idx] == colours[rw_idx] == colours[lf_idx] == colours[rf_idx]:
                target_colour = colours[lw_idx] # this is the colour of the holds the user is going to climb
                # get the indexes of all the same colours
                coloured_idx = [idx for idx,colour in enumerate(colours) if colour == target_colour]
                # keep all same coloured holds
                coloured_holds = [contours[x] for x in coloured_idx]

                logging.info("User has both hands and feet on same coloured holds. LET'S GO!!!") 
                return [target_colour, coloured_holds]
            
            # if colours not same, we cannot start yet
            else:
                logging.info("User has both hands and feet on holds, but they are not of same colour.")
                return False 
        
        except:
            logging.info("ValueError.")
            return False


# Function to get the length of the user's arm during start position
# This is used to ensure that our relative distance calculation later on is accurate no matter how far the camera is from the wall
def getArmLength(pose_results):
    l_wrist = pose_results['l_wrist']
    l_elbow = pose_results['l_elbow']
    r_wrist = pose_results['r_wrist']
    r_elbow = pose_results['r_elbow']

    # calculate arm length (elbow to wrist)
    # we take elbow - wrist because origin is on top 
    l_arm_length = abs(math.hypot(l_elbow[0] - l_wrist[0], l_elbow[1] - l_wrist[1]))
    r_arm_length = abs(math.hypot(r_elbow[0] - r_wrist[0], r_elbow[1] - r_wrist[1]))

    # return the longest arm length as it is the most accurate
    return round(max(l_arm_length, r_arm_length), 2)
   


# Function to check which limbs are not on holds
# Since 3-point contact is required, there should only be 1 limb off the hold at all times
# This will be the limb used to look for the next hold to move to
def checkLimbs(contours, poses, arm_length, threshold):
    on_hold = []

    # keep only the poses that requires touch on holds (wrists and feet)
    required_poses = dict((k, poses[k]) for k in ('l_wrist', 'r_wrist', 'l_foot', 'r_foot'))

    for part in required_poses:
        # check if the part is touching a hold
        part_condition = holdTouchCondition(contours, poses[part], arm_length, threshold)

        if part_condition[1] == "touched":
            on_hold.append(part)
    
    # all limbs on holds, need to lift 1 off
    if len(on_hold) == 4:
        logging.info("Lift 1 limb off a hold to start voice recommender.")
        return False, len(on_hold)

    # only 1 limb off the hold, 3-point contact good to go
    elif len(on_hold) == 3:
        logging.info("Only 1 limb off the hold, start voice recommender.")
        return [part for part in required_poses if part not in on_hold][0], len(on_hold) # return the body part that is off the wall
    
    # too many limbs off the holds
    else:
        logging.info("Too many limbs off holds, keep 3-point contact.")
        return False, len(on_hold)
    



# Function to get the nearest hold to the required body part
def nearestHold(contours, pose_result, part_name, arm_length, threshold, text):
    part = pose_result[part_name]
    holds = []

    # get the coordinates of the contour of each hold
    for contour in contours:
        # find the highest and lowest point of each contour
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        btm = tuple(contour[contour[:, :, 1].argmax()][0])

        # save to a list of dictionaries
        holds.append({'contour': contour, 'top': top, 'bottom': btm})


    # get a list of contours that the body part is outside of
    contours_outside = [hold for hold in holds if cv2.pointPolygonTest(hold['contour'], part, False) == -1]


    for contour in contours_outside:
        # calculate the nearest distance from the part to any point on the edge of the contour
        dist = cv2.pointPolygonTest(contour['contour'], part, True)
        # get the relative absolute distance
        dist = round((abs(dist) / arm_length), 2)

        # save additional distance key to the current dictionary
        contour['distance'] = dist


    # keep only holds greater than threshold -- considered outside
    holds_outside = [hold for hold in contours_outside if hold['distance'] > threshold]
    # text.append(f"length before doing y-manipulation: {len(holds_outside)}")


    # Keep holds for part if Y-coordinate for hold is less than Y-coordinate for part
    # I am using less than because in OpenCV, (0,0) starts on the top left side
    # Means the hold must be higher than the part
    if 'wrist' in part_name:
        viable_holds = [hold for hold in holds_outside if hold['bottom'][1] <= part[1]]


    # for foot it is a little bit different
    # the hold has to be higher than the foot but lower than the elbow
    elif 'foot' in part_name:
        if part_name.startswith('l'):
            viable_holds = [hold for hold in holds_outside if (hold['bottom'][1] <= part[1]) and (hold['top'][1] >= pose_result['l_elbow'][1])]
        elif part_name.startswith('r'):
            viable_holds = [hold for hold in holds_outside if (hold['bottom'][1] <= part[1]) and (hold['top'][1] >= pose_result['r_elbow'][1])]


    if (len(viable_holds) > 0):
        # sort the holds and get the nearest to the part, and extract their contour coordinates
        nearest_hold = sorted(viable_holds, key=lambda i: i['distance'])[0]['contour']

    else:
        nearest_hold = False
        logging.info(f"There is no next hold for {part_name}.")
        text.append(f"There is no next hold for {part_name}.")

    return nearest_hold, text



# Function to calculate angle between 2 points
def calculateAngle(p1, p2):
    # Difference in x coordinates
    dx = p1[0] - p2[0]

    # Difference in y coordinates
    dy = p1[1] - p2[1]

    # calculate angle between 2 points and convert to degrees
    result = math.degrees(math.atan2(dy, dx))

    return result



# Function to calculate relative distance for part to the nearest hold
def calculateRelativeDistance(pose_results, hold, part_name, arm_length):
    part = pose_results[part_name]

    # calculate the nearest distance from the part to any point on the edge of the contour
    dist = cv2.pointPolygonTest(hold, part, True)

    # calculate relative distance from wrist to hold vs arm length
    relative_dist = round(abs(dist) / arm_length, 2)
    
    return relative_dist



# Function to draw outputs on image frame before start of run
def drawOutputsBeforeRun(image, contours, pose_results, colour_values, colour_list, text):
    output_img = image.copy()

    # draw visualisations for human pose parts
    for part in pose_results:
        cv2.circle(output_img, pose_results[part], 3, (0,255,0), 3)

    # draw visualisations for contours in their respective colours
    for idx, contour in enumerate(contours):
        cv2.drawContours(output_img, [contour], -1, colour_values[colour_list[idx]], 3)

    # draw text for angles and distances on output frame
    y_start = 600
    y_increment = 30
    for i, line in enumerate(text):
        y = y_start + i*y_increment
        cv2.putText(output_img, line, org=(20, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), 
        thickness=3)

    return output_img



# Function to draw outputs on image frame
def drawOutputs(image, contours, colour, colour_values, pose_results, text, nearest_hold=False):
    output_img = image.copy()

    # draw the contours we have obtained from our model with the correct colours
    for idx, contour in enumerate(contours):
        cv2.drawContours(output_img, [contour], -1, colour_values[colour], 3)

    # draw circles on all parts
    for part in pose_results:
        cv2.circle(output_img, pose_results[part], 3, (0,255,0), 3)

    # draw circles on the nearest hold
    if type(nearest_hold) != bool:
        cv2.drawContours(output_img, [nearest_hold], -1, (247,255,4), 3) # use turquoise to highlight nearest contour

    # draw text for angles and distances on output frame
    y_start = 600
    y_increment = 30
    for i, line in enumerate(text):
        y = y_start + i*y_increment
        cv2.putText(output_img, line, org=(20, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), 
        thickness=3)

    return output_img



# Based on the coordinates of the top of each hold, get the highest one
def getLastHold(contours):
    holds = []
    # get the coordinates of the contour of each hold
    for contour in contours:
        # find the highest and lowest point of each contour
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        btm = tuple(contour[contour[:, :, 1].argmax()][0])

        # save to a list of dictionaries
        holds.append({'contour': contour, 'top': top, 'bottom': btm})
    
    # get the contour of the highest hold (lowest y)
    highest_hold = sorted(holds, key=lambda i: i['top'][1])[0]['contour']
    return highest_hold



# Function to check if the run has ended or not
# End condition = both hands touching the highest hold
def checkEndCondition(highest_hold, pose_results, arm_length, threshold):
    l_wrist = pose_results['l_wrist']
    r_wrist =pose_results['r_wrist']

    l_touched, r_touched = False, False

    # first check if the left and right wrist is inside or touching any contour
    # if the wrist is not outside the hold contour, means it is touching
    if cv2.pointPolygonTest(highest_hold, l_wrist, False) != -1:
        l_touched = True
    if cv2.pointPolygonTest(highest_hold, r_wrist, False) != -1:
        r_touched = True

    # within a certain threshold, we also take it as the part is touching the hold
    # calculate the nearest distance from the part to any point on the edge of the contour
    l_dist = abs(cv2.pointPolygonTest(highest_hold,l_wrist,True))
    l_dist = round((l_dist / arm_length), 2)
    r_dist = abs(cv2.pointPolygonTest(highest_hold,r_wrist,True))
    r_dist = round((r_dist / arm_length), 2)

    if l_dist < threshold:
        l_touched = True
    if r_dist < threshold:
        r_touched = True

    # if both wrists are touching the last hold, we end the programme
    if (l_touched == True) and (r_touched == True):
        return True
    # else continue as per normal
    else:
        return False

    
def write2stream(part, relative_dist, stream, MAX_DIST=7, MIN_DIST=0.6, DURATION=0.1, SOUND_ARGS=(333, 450), VOL_SCALE=0.2, FS=44100):
    """Writes to a pyaudio output stream to generate sound feedback 

    Args:
        part (str): [l_foot | l_hand | r_foot | r_hand]
        relative_dist (float): relative distance wrt arm length
        stream (pyaudio.Pyaudio.Stream): output stream
        MAX_DIST (int, optional): maximum expected relative_dist. Defaults to 7 (statistical maximum).
        MIN_DIST (float, optional): minimum expected relative_dist. Defaults to 0.6 (statistical minimum).
        DURATION (float, optional): Buffer duration of sound (s). Defaults to 0.1.
        SOUND_ARGS (tuple, optional): Non-linear mapping of relative_dist to sound. Mapping: freq(relative_dist) = 333/relative_dist + 450.
        VOL_SCALE (float, optional): Scale factor for volume. Defaults to 0.2.
        FS (int, optional): Audio sampling frequency. Defaults to 44100.
    """
    if relative_dist != None and part != None:
        side = part[0]
        if relative_dist > MIN_DIST:
            if relative_dist > MAX_DIST:
                relative_dist = MAX_DIST
            freq = int(SOUND_ARGS[0]/relative_dist + SOUND_ARGS[1])
            channel_mask = [side == "l", side == "r"] * int((FS*DURATION)) # generate mask to fire 1 channel, clear other
            time_samples = np.arange(FS * 2*DURATION) * freq / FS
            samples = (np.sin(2 * np.pi * time_samples*channel_mask)).astype(np.float32)
        
            samples *= VOL_SCALE
            output_bytes = samples.tobytes()
            # stream.write(output_bytes)
            return output_bytes

def beepSound(stream, DURATION=0.3, FREQ_RANGE=(300,700), VOL_SCALE=0.2, FS=44100):
    """Fast beeps when part is near next hold

    Args:
        part (str): [l_foot | l_hand | r_foot | r_hand]
        stream (pyaudio.Pyaudio.Stream): output stream
        DURATION (int, optional): Duration of beep (s). Defaults to 1.
        FREQ_RANGE (tuple, optional): Range of values of frequency (Hz). Defaults to (300,700).
        VOL_SCALE (float, optional): Scale factor for volume. Defaults to 0.2.
        FS (int, optional): Audio sampling frequency. Defaults to 44100.
    """
    (LOW_FREQ, HIGH_FREQ) = FREQ_RANGE
    time_samples1 = np.arange(FS * 2*(1/4*DURATION)) * LOW_FREQ / FS
    time_samples2 = np.arange(FS * 2*(1/4*DURATION)) * HIGH_FREQ / FS
    time_samples = np.tile(np.concatenate((time_samples1, time_samples2), axis=0), 2)
    samples = (np.sin(2 * np.pi * time_samples)).astype(np.float32)
    ones = np.ones(int(0.2*len(samples)))
    zeros = np.zeros(int(0.05*len(samples)))
    audio_mask = np.tile(np.concatenate((ones, zeros), axis=0), 4)
    samples *= audio_mask
        
    samples *= VOL_SCALE
    output_bytes = samples.tobytes()
    return output_bytes


# Function to post data to DynamoDB
def postRequest(color):
    url = "https://mzpoqw4tt9.execute-api.ap-southeast-1.amazonaws.com/Prod/applewatchstats"

    random_id = random.randint(10000000, 99999999)
    string_id = str(random_id)

    now = datetime.datetime.now()
    formatted_date = now.strftime("%d/%m/%y")

    data = {
        "id": string_id,
        "color": color,
        "timeInSeconds": "empty",
        "date": formatted_date,
        "calories": "empty",
        "heartRate": "empty"
    }

    response = requests.post(url, json=data)

    print(requests)
    print(response.status_code)
    print(response.json())