# imports
import cv2
import mediapipe as mp
import numpy as np
from scipy import ndimage

import time
import subprocess
import json
import shutil
import os
import datetime
import serial
import math
import copy
from informativeDrawings import informativeDrawings
from traceSkeleton import trace_skeleton
import gridCreation
import imageOperations
import exportOperations

sine_pattern = None

##################
# global variables
##################

# person detection and automatic drawing
time_last_detection = time.time()
time_delay_photo = 8
autodraw = False

# script options
run_params = {
    "scripts": True,
    "print": True,
    "mergedSvg": True
}

states = ["cam","shooting","image_select","style_select","export"]
current_state = "cam"
image_container = []
line_container = []

# params for image size and position in grid
grid = []

# a mask image that is used to avoid front portraits
# to be overdrawn by back portraits (which are drawn later)
mask_img = {}

# the preview image
preview_img = {}

# background image for image subtraction
bg_image = None
bg_image_inv = None

# face detection and autodraw
is_face_detected = False

# calculation and config params
layout = {}

f = open("config.json")
config = json.load(f)

# filter variables
line_size = 11
blur_edge = 5

blur_image = 8
circle_radius = 1.0
circle_border = 56*2 +1
clip_limit = 2
render_quality = 512

d_line = 10
angle_line = 45
threshold_min = 128
threshold_Max = 200


# rotation of cam image
cam_img_rotate = 3  # start with 90° counterclockwise. press R to rotate
img_rotater = ["", cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]

show_fps = False  # fps counter
canvas_full = False  # is the Canvas fully filled?


#########
# functions
#########
def show_usage():
    print("    /|    //| |                                                      /__  ___/                                                 ")
    print("   //|   // | |     ___       __      ___     ( )  ___      __         / /   __      ___      ___      ___       __    __  ___ ")
    print("  // |  //  | |   //   ) ) //   ) ) ((   ) ) / / //___) ) //  ) )     / /  //  ) ) //   ) ) //   ) ) //   ) ) //   ) )  / /    ")
    print(" //  | //   | |  //   / / //   / /   \ \    / / //       //          / /  //      //   / / //       //   / / //   / /  / /     ")
    print("//   |//    | | ((___/ / //   / / //   ) ) / / ((____   //          / /  //      ((___( ( ((____   ((___( ( //   / /  / /      ")
    print("Press [p] for taking a picture")
    print("Press [ ] to take a photo")
    print("Press [#] for fastlane export")
    print("Press [f] for a FPS counter in the console")
    print("Press [g] for export a test grid")
    print("Press [ESC] to quit")


def create_line_variations(image_container,line_container):
    blurs = [0,3,5,11]
    for i in range(len(image_container)):
        if i>0:
            image_container[i] = cv2.GaussianBlur(image_container[i], (blurs[i],blurs[i]), 0)
        line_container.append(imageOperations.extract_contours(image_container[i]))
        image_container[i][:] = 255
        image_container[i] = imageOperations.create_preview_from_polys(line_container[i], image_container[i],(0,0,0),1)
    preview = np.concatenate((image_container), axis=1)
    cv2.putText(preview,"Bild auswaehlen (n abbruch)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0)
    for i in range(len(image_container)):
        cv2.putText(preview,str(i), (20 + i*image_container[0].shape[1],100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
    cv2.imshow("tracant", preview)

def finalize_image(output_image,lines):
    
    f = open("layer.json")
    s = json.load(f)

    # export
    dx = -20
    dy = 0
    wp = 280
    hp = 400

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

    #lines = imageOperations.extract_contours(output_image)
    preview = output_image.copy()
    

    preview[:] = 255
        
    for i,item in enumerate(s["hatchNoise"]):
        h = imageOperations.img_to_polys(output_image,item)
        exportOperations.export_polys_as_hpgl(h, output_image.shape, timestamp +str(i) + "_"+ item["color"], wp, hp,"70_90",dx,dy)
        preview = imageOperations.create_preview_from_polys(h, preview,(128,128,128),1)
        
    exportOperations.export_polys_as_hpgl(lines, output_image.shape, timestamp + "4_black", wp, hp,"70_90",dx,dy,True)
    preview = imageOperations.create_preview_from_polys(lines, preview,(0,0,0),1)
    cv2.putText(preview,"Leertaste fuer neue Aufnahme", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0)
    cv2.imshow("tracant", preview)

def update_navigation(key,current_state,image_container,line_container):
    if current_state == "image_select" or current_state == "style_select":
        temp = image_container[0]
        for i in range(len(image_container)):
            image_container[i] = temp.copy()
        if current_state == "image_select":
            current_state = "style_select"
            line_container = []
            create_line_variations(image_container,line_container)
        elif current_state == "style_select":
            finalize_image(output_image,line_container[key])
            current_state = "export"
    return current_state,line_container

##############
# main script
##############

# init mediapipe detectors for face and background subtraction
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# create grid
layout = gridCreation.create_grid(config)

# show usage/documentation in console
show_usage()

# select capture device
cap = cv2.VideoCapture(config["hardware"]["camId"])

# capture loop
while cap.isOpened():
    # count fps
    start_time = time.time()  # start time of the loop

    if current_state == "cam" or current_state == "shooting":
        # read image
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        #image= cv2.imread("test.jpg")
        image = cv2.rotate(image, img_rotater[1])

        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rotate image since camera is rotated as well
        
    # if cam_img_rotate != 0:
    #     image = cv2.rotate(image, img_rotater[cam_img_rotate])

        # init output image with zeros
        output_image = np.zeros(
            (layout["grid"][0]["heightPx"], layout["grid"][0]["widthPx"], 3), np.uint8,)

        # calulate image ratio and resize image to config settings
        ratio = image.shape[0] / image.shape[1]

        # landscape
        # image = cv2.resize(image, (int(config["resImg"] / ratio), config["resImg"]))

        # portrait
        image = cv2.resize(image, (int(config["layout"]["resImg"] / ratio), config["layout"]["resImg"]))

        # set not writable for more performance
        image.flags.writeable = False

        # detect faces
        faces = face_detection.process(image)

        # continue if face detected
        if faces.detections:
            if not is_face_detected:
                time_last_detection = time.time()

            is_face_detected = True

            # get segmented image
            segmentation_bg = selfie_segmentation.process(image)

            # prepare and cut out img
            #condition = np.stack((segmentation_bg.segmentation_mask,) * 3, axis=-1) > 0.1
            condition = np.stack((segmentation_bg.segmentation_mask,) * 3, axis=-1)
            
            # init bg image
            if bg_image is None:
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = (255, 255, 255)

            if bg_image_inv is None:
                bg_image_inv = np.zeros(image.shape, dtype=np.uint8)
                bg_image_inv[:] = (0, 0, 0)

            
            out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            

            # blur part around face
            out_blur = out
            #out_blur = cv2.medianBlur(out, blur_image)
            circular_mask = np.zeros(out.shape, dtype=np.uint8)

            for face in faces.detections:
                bb = face.location_data.relative_bounding_box
                c_x = int((bb.xmin + 0.5*bb.width)*out.shape[1])
                c_y = int((bb.ymin + 0.5*bb.height)*out.shape[0])
                r = int(max(bb.height*out.shape[0], bb.width*out.shape[1])*0.5*circle_radius)
                cv2.circle(circular_mask, (c_x, c_y), r, (255,255,255), thickness=-1)

            circular_mask = cv2.GaussianBlur(circular_mask, (circle_border, circle_border), 0)
            circular_mask = circular_mask/255

            out =  (out * circular_mask + out_blur * (1 - circular_mask)).astype(np.uint8)

            # remove bg from edge image
            output_image =  (out * condition + bg_image * (1 - condition)).astype(np.uint8)
            output_image_inv =  (out * condition + bg_image_inv * (1 - condition)).astype(np.uint8)
    

            # show output image
            if current_state == "cam" :
                cv2.imshow("tracant", output_image)
            elif current_state == "shooting" :
                image_container.append(output_image.copy())
                if len(image_container)>=4:
                    current_state = "image_select"
                    preview = np.concatenate((image_container), axis=1)
                    cv2.putText(preview,"Bild auswaehlen(n abbruch)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0)
                    for i in range(len(image_container)):
                        cv2.putText(preview,str(i), (20 + i*image_container[0].shape[1],100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
                    cv2.imshow("tracant", preview)
        
        
        
        #imageOperations.trace_image(img, export_preview)

        
        
    else:
        is_face_detected = False

    # get last pressed key
    k = cv2.waitKey(1)
    if k == -1:
        pass
    elif k == ord("f"):
        show_fps = not show_fps
    elif k == ord(" "):
        if current_state == "cam":
            current_state = "shooting"
            image_container = []
        elif current_state == "export":
            current_state = "cam"
    elif k == ord("n"):
        current_state = "cam"
    elif k == ord("0"):
        current_state,line_container = update_navigation(0, current_state, image_container, line_container)
    elif k == ord("1"):
        current_state,line_container = update_navigation(1, current_state, image_container, line_container)
    elif k == ord("2"):
        current_state,line_container = update_navigation(2, current_state, image_container, line_container)
    elif k == ord("3"):
        current_state,line_container = update_navigation(3, current_state, image_container, line_container)

    elif k == ord("#"):
        f = open("layer.json")
        s = json.load(f)

         #7576a
        '''
        dx = -20
        dy = 0
        wp = 280
        hp = 400
        '''
                
        #7475a
        dx = 163.5
        dy = 220
        wp = 177
        hp = 350

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

        lines = imageOperations.extract_contours(output_image)
        preview = output_image.copy()

        preview[:] = 255
        
        for i,item in enumerate(s["hatchNoise"]):
            h = imageOperations.img_to_polys(output_image,item)
            exportOperations.export_polys_as_hpgl(h, output_image.shape, timestamp +str(i) + "_"+ item["color"], wp, hp,"70_90",dx,dy)
            preview = imageOperations.create_preview_from_polys(h, preview,(128,128,128),1)
        
        exportOperations.export_polys_as_hpgl(lines, output_image.shape, timestamp + "4_black", wp, hp,"70_90",dx,dy,False)
        preview = imageOperations.create_preview_from_polys(lines, preview,(0,0,0),1)
        cv2.imshow("preview", preview)
        
    elif k == ord("g"):
        f = open("layer.json")
        s = json.load(f)

        # export
        #7576a
        '''
        dx = -20
        dy = 0
        wp = 280
        hp = 400
        '''
                
        #7475a
        dx = 163.5
        dy = 220
        wp = 177
        hp = 350

        p = [[(0, 0), (2, 2), (2, 2), (4, 4)]]
        
        exportOperations.export_polys_as_hpgl(p, output_image.shape, "grid", wp, hp,"70_90",dx,dy,False)


    elif k & 0xFF == 27:
        break

    if (show_fps):
        print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


cap.release()
cv2.destroyAllWindows()
