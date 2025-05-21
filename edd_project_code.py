import os
import subprocess

if not os.path.exists("sort"):
    subprocess.run(["git", "clone", "https://github.com/abewley/sort"], check=True)

sort_file = "sort/sort.py"
with open(sort_file, "r") as file:
    content = file.read()

if "TkAgg" in content:
    content = content.replace("TkAgg", "Agg")
    with open(sort_file, "w") as file:
        file.write(content)


import ast
import cv2
import numpy as np
import os
import pandas as pd
from roboflow import Roboflow
import supervision as sv
import string
import easyocr
import re
import time
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
from ultralytics import YOLO
from sort.sort import *

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

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
                                                            True,#TODO
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




results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

rf = Roboflow(api_key="6nor1stR1ivUYVOjcIvs")
project = rf.workspace().project("handicap-placard-detection")
handicap_detector = project.version(1).model

# load video
cap = cv2.VideoCapture(0) #TODO
cap.set(3, 1080)
cap.set(4, 1080)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or *'avc1'
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1080, 1080))

capture_duration = 10

start_time = time.time()
while int(time.time() - start_time) < capture_duration:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 0)
        out.write(frame)
    else:
        break

cap.release()
out.release()

cap = cv2.videoCapture("testing_vids/output.mp4")

vehicles = [2, 3, 5, 7]

frame_height = 1080
frame_width = 1080
# read frames
frame_nmr = -1
ret = True
handicap_found = False
plate_cache = {}

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    margin = 50
    if not ret:
        break

    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    track_ids = mot_tracker.update(np.asarray(detections_)) if len(detections_) > 0 else np.empty((0, 5))

    license_plates = license_plate_detector(frame)[0]
    for lp in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            if license_plate_text:
                plate_cache[car_id] = {
                    'frame': frame_nmr,
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    },
                    'car': {
                        'bbox': [xcar1, ycar1, xcar2, ycar2]
                    }
                }


    handicap_result = handicap_detector.predict(frame).json()
    handicap_detections = sv.Detections.from_inference(handicap_result)
    if handicap_detections.xyxy.any():
        x1, y1, x2, y2 = handicap_detections.xyxy[0]
        if all([x1 > 50, x2 < frame_width - margin, y1 > 50, y2 < frame_height - margin]):
            print(f'Handicap placard detected at Frame {frame_nmr}')
            handicap_found = True

results = {}
if not handicap_found:
    for car_id, data in plate_cache.items():
        frame = data['frame']
        if frame not in results:
            results[frame] = {}
        results[frame][car_id] = {
            'car': data['car'],
            'license_plate': data['license_plate']
        }


write_csv(results, f'/content/drive/MyDrive/EDD_Project/testnew.csv')
