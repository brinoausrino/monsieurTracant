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
circle_border = 56
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

ser = 0
try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
except:
    print("LED not available")


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
    print("Press [s] to start a new image after the current one was full")
    print("Press [q] to force a reset and start drawing a new image ")
    print("Press [r] for rotating the cam - not functional yet")
    print("Press [f] for a FPS counter in the console")
    print("Press [g] for export a test grid as svg")
    print("Press [ESC] to quit")

# gui callback functions
def set_line_size(x):
    global line_size
    line_size = x*2+3


def set_blur_edge(x):
    global blur_edge
    blur_edge = x*2+1

def set_blur_image(x):
    global blur_image
    blur_image = x*2+1

def set_circle_border(x):
    global circle_border
    circle_border = x*2+1

def set_circle_radius(x):
    global circle_radius
    circle_radius = x/10

def set_clip_limit(x):
    global clip_limit
    clip_limit = x

def set_d_line(x):
    global d_line
    d_line = x

def set_angle_line(x):
    global angle_line
    angle_line = x

def set_threshold_line(x):
    global threshold_min
    threshold_min = x

def set_threshold_line2(x):
    global threshold_Max
    threshold_Max = x


def start_new_drawing():
    # clean up all files from /out/ into a timestamped subfolder
    target_dir = os.path.join(os.getcwd(), config["outputFolder"], datetime.datetime.now().strftime('%y%m%d_%H%M'))
    os.makedirs(target_dir)

    file_names = os.listdir(config["outputFolder"])

    for file_name in file_names:
        if "." in file_name:  # move only files
            shutil.move(os.path.join(config["outputFolder"]+"/", file_name), target_dir)

    # reset the mask and preview by creating a new grid
    grid = []
    gridCreation.create_single(config)

    # reset the currentId JSON counter to 0
    f = open("counter.json")
    cf = json.load(f)
    cf["currentId"] = 0
    with open("counter.json", "w") as f:
        json.dump(cf, f)

def page_is_full():
    # code for having a full page, currently happening
    # when counter.json has counterId > 37
    # we check if the export_image call failed because of this issues
    f = open("counter.json")
    cf = json.load(f)
    currentId = cf["currentId"]

    # if this not the bug we will stop here
    if currentId <= 36:
        print("An exception occured during export_image function.")
        return

    print("I have filled the paper completely.")
    print("I will now sign it and then you can place a new canvas in the plotter.")
    print("Press [s] when you are ready.")
    # otherwise the paper is fully filled and we should
    # (a) wait for a new paper to be inserted, prevent drawing in the mean time
    canvas_full = True  # this does not work if its only set inside this function. we set it before calling this function as well

    # (b) sign the document with an pre-made svg signature


# mask image, update image mask and export image to hpgl
def export_image(output_image, mask):
    global run_params
    global mask_img
    global preview_img
    global config

    # get current position in grid from counter.json
    f = open("counter.json")
    cf = json.load(f)
    currentId = cf["currentId"]

    # get size and position settings for current element
    elemSettings = grid[currentId]

    # rotate images
    output_image = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    x_px = elemSettings["xPx"]
    y_px = elemSettings["yPx"]
    w_px = elemSettings["widthPx"]
    h_px = elemSettings["heightPx"]

    # scale and translate mask for mask_img
    mask_scaled_size = (w_px, h_px)
    mask = cv2.resize(mask, mask_scaled_size)

    # extract t_mask from mask_img to mask image
    t_mask = mask_img[y_px: y_px + h_px, x_px: x_px + w_px]
    t_mask = cv2.resize(t_mask, (output_image.shape[1], output_image.shape[0]))
    t_mask = np.invert(t_mask)

    # convert output image to grayscale and mask it
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    output_image = np.maximum(output_image, t_mask)

    # create and show preview image
    # resize the output_image, merge it with matiching section of preview image
    # and insert merged image
    preview_img[y_px: y_px + h_px, x_px: x_px + w_px] = np.minimum(
        preview_img[y_px: y_px + h_px, x_px: x_px + w_px],
        cv2.resize(output_image, mask_scaled_size)
    )
    cv2.imshow("preview_img", preview_img)

    # update mask with current image
    mask_img[y_px: y_px + h_px, x_px: x_px + w_px] = np.minimum(
        mask, mask_img[y_px: y_px + h_px, x_px: x_px + w_px]
    )

    # export image and mask
    cv2.imwrite(config["outputFolder"] + "preview.bmp", preview_img)
    cv2.imwrite(config["outputFolder"] + "mask.bmp", mask_img)
    cv2.imwrite(config["outputFolder"] + str(currentId) + ".bmp", output_image)

    # run vpype scripts to trace image, position on paper, export as hpgl and plot
    ser.write('h'.encode())
    if run_params["scripts"]:
        # trace
        bashCommand = "sh 0_trace_image.sh " + config["outputFolder"] + str(currentId)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # place vector image on paper
        bashCommand = (
            "sh 1_scale_position.sh " + config["outputFolder"] + str(currentId) + " "
            + str(elemSettings["width"])
            + "cm "
            + str(elemSettings["height"])
            + "cm "
            + str(elemSettings["x"])
            + "cm "
            + str(elemSettings["y"])
            + "cm"
        )
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # merge in svg for testing purposes
        # !caution: takes some time
        if run_params["mergedSvg"]:
            bashCommand = "sh 4_mergeLayers.sh " + config["outputFolder"] + str(currentId) + "_scaled"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        # export as hpgl
        bashCommand = "sh 3_export_hpgl.sh " + config["outputFolder"] + str(currentId) + "_scaled"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # print hpgl
    if run_params["print"]:
        bashCommand = "python hp7475a_send.py " + config["outputFolder"] + str(currentId) + "_scaled.hpgl -p " + config["hardware"]["plotterSerial"]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # update id and write to file
    cf["currentId"] += 1
    with open("counter.json", "w") as f:
        json.dump(cf, f)

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
        output_image = image_container[0]
        for i in range(len(image_container)):
            image_container[i] = output_image.copy()
        if current_state == "image_select":
            current_state = "style_select"
            line_container = []
            create_line_variations(image_container,line_container)
        elif current_state == "style_select":
            finalize_image(image_container[0],line_container[0])
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

# create openCV window with sliders for edge filter
cv2.namedWindow("output")
cv2.namedWindow("cam")

cv2.createTrackbar("blur_image", "cam", 0, 40, set_blur_image)
cv2.createTrackbar("circle_border", "cam", 22, 80, set_circle_border)
cv2.createTrackbar("circle_radius", "cam", 10, 30, set_circle_radius)
cv2.createTrackbar("clip_limit", "cam", 2, 10, set_clip_limit)

cv2.namedWindow("output")
cv2.createTrackbar("d_line", "output", 12, 100, set_d_line)
cv2.createTrackbar("angle_line", "output", 45, 180, set_angle_line)
cv2.createTrackbar("threshold", "output", 0, 254, set_threshold_line)
cv2.createTrackbar("threshold2", "output", 200, 245, set_threshold_line2)

# init info drawings
# opt = informativeDrawings.Options()
# opt.dataroot = "test.png"
# informativeDrawings.init_nn(opt)

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

    # autodraw
    if autodraw and is_face_detected and time.time() - time_last_detection > time_delay_photo:
        try:
            export_image(output_image, mask)
        except:
            canvas_full = True  # prevent further drawing till [s] is pressed
            page_is_full()

        time_last_detection = time.time()

    # get last pressed key
    k = cv2.waitKey(1)
    if k == -1:
        pass
    # if key was 'p' then start exporting and plotting
    elif k == ord("p") and not canvas_full:
        try:
            export_image(output_image, mask)
        except:
            canvas_full = True  # prevent further drawing till [s] is pressed
            page_is_full()
    # if key was 'r' then rotate image
    elif k == ord("r"):
        cam_img_rotate = (cam_img_rotate+1) % 4  # select next rotation style (none, 90ccw,180,90cw)
    # if key was 'f' then to show FPS counter in console
    elif k == ord("f"):
        show_fps = not show_fps
    # if key was 's' and  to show FPS counter in console
    elif k == ord("s"):
        if canvas_full:
            canvas_full = False
            start_new_drawing()
            print("I will now draw again. Thanks for the paper")
        else:
            print("The image was not full yet. I will continue drawing on the old one.")
    elif k == ord("q"):
        start_new_drawing()
        print("Hey! I still wanted to paint that! Grrml. Ok i will start a new one.")

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

        # export
        dx = -20
        dy = 0
        wp = 280
        hp = 400

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

        lines = imageOperations.extract_contours(output_image)
        preview = output_image.copy()

        preview[:] = 255
        
        for i,item in enumerate(s["hatchNoise"]):
            h = imageOperations.img_to_polys(output_image,item)
            exportOperations.export_polys_as_hpgl(h, output_image.shape, timestamp +str(i) + "_"+ item["color"], wp, hp,"70_90",dx,dy)
            preview = imageOperations.create_preview_from_polys(h, preview,(128,128,128),1)
        
        exportOperations.export_polys_as_hpgl(lines, output_image.shape, timestamp + "4_black", wp, hp,"70_90",dx,dy,True)
        preview = imageOperations.create_preview_from_polys(lines, preview,(0,0,0),1)
        cv2.imshow("preview", preview)

    elif k & 0xFF == 27:
        break

    if (show_fps):
        print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


cap.release()
cv2.destroyAllWindows()
