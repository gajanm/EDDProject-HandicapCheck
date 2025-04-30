
#git clone https://github.com/abewley/sort

#pip install -r EDDProject-HandicapCheck/requirements.txt

#sed -i 's/TkAgg/Agg/' sort/sort.py

import ast
import cv2
import numpy as np
import pandas as pd
import os
import pandas as pd
from roboflow import Roboflow
import supervision as sv
import string
import easyocr
import re

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'handicap_detected', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()




# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '2': 'Z',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """

    print(f"Extracted Text: {text}")



    if len(text)!=6:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_int_to_char,
               4: dict_int_to_char, 5: dict_int_to_char}
    for j in [0, 1, 2, 3, 4, 5]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)


    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        text = re.sub(r'[^\w\s]', '', text)

        if license_complies_format(text):
            return format_license(text), score

    return None, None



def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1



os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

from ultralytics import YOLO
import cv2

from sort.sort import *

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('/content/drive/MyDrive/EDD_Project/license_plate_detector.pt')

rf = Roboflow(api_key="6nor1stR1ivUYVOjcIvs")
project = rf.workspace().project("handicap-placard-detection")
handicap_detector = project.version(1).model

# load video
cap = cv2.VideoCapture('/content/drive/MyDrive/EDD_Project/new_vid.mp4')

vehicles = [2, 3, 5, 7]

frame_height = 980
frame_width = 884
# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    margin = 50
    if ret:
        results[frame_nmr] = {}
        handicap_result = handicap_detector.predict(frame).json()
        handicap_detections = sv.Detections.from_inference(handicap_result)
        if handicap_detections.xyxy.any():
          x1, y1, x2, y2 = handicap_detections.xyxy[0]
          if x1>50 and x1<frame_width-margin and y1>50 and y1<frame_height-margin and x2>50 and x2<frame_width-margin and y2>50 and y2<frame_height-margin:
            print(f'Handicap detected at Frame {frame_nmr}')
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                  detections_.append([x1, y1, x2, y2, score])
            if len(detections_) > 0:
                track_ids = mot_tracker.update(np.asarray(detections_))
            else:
                track_ids = np.empty((0, 5))



            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            print(f"Frame {frame_nmr} license plates: {license_plates.boxes.data.tolist()}")

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                print(f"License plate matched to car ID: {car_id}")


                if car_id != -1:

                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]


                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    cv2.imwrite(f'/content/frame_{frame_nmr}_plate.png', license_plate_crop_thresh)

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                    print(f"Read license plate: {license_plate_text}, Score: {license_plate_text_score}")

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

print(results)
write_csv(results, '/content/drive/MyDrive/EDD_Project/test2.csv')
