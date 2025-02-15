# imports
import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import json
import shutil
import os, datetime
import serial

ser = 0
try:
    ser = serial.Serial('/dev/ttyACM0',9600, timeout=1)
except:
    print("LED not available") 

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

# folder to put files
output_folder = "out/"

# params for image size and position in grid
grid = []

# a mask image that is used to avoid front portraits
# to be overdrawn by back portraits (which are drawn later)
mask_img = {}

# the preview image
preview_img = {}

# background image for image subtraction
bg_image = None

# face detection and autodraw
is_face_detected = False

# calculation and config params
cm_to_px = 1
center_px = [0, 0]
config = {}

f = open("hardwareConfig.json")
hardware_config = json.load(f)

# filter variables
line_size = 11
blur_edge = 5

blur_image = 8
circle_radius = 1.0
circle_border = 56
clip_limit = 2

# rotation of cam image
cam_img_rotate = 1  # start with 90° counterclockwise. press R to rotate
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

# filter edge detection
def edge_mask(img, line_size, blur_value):

    gray_blur = cv2.medianBlur(img, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        line_size,
        blur_value,
    )
    return edges

# filter edge detection
def pencil_sketch(img, line_size, blur_value):

    # invert
    img_invert = cv2.bitwise_not(img)
    # blur
    img_smoothing = cv2.GaussianBlur(img_invert, (blur_value*2+1, blur_value*2+1), sigmaX=0, sigmaY=0)
    # subtract
    final = cv2.divide(img, 255 - img_smoothing, scale=255)
    final, mask = cv2.threshold(final, 220, 255, cv2.THRESH_BINARY)
    return final

# creates person grid based on config file
# some calculations me be confuding, since the final image is rotated 90°
# therefore x and y are swapped
def create_grid(grid):
    global cm_to_px
    global mask_img
    global preview_img
    global center_px
    global config

    # load config file
    f = open("gridConfig.json")
    config = json.load(f)

    # printable length on paper 
    printableLengthPaperY = (
        config["paperSize"][1] - config["margin"][0] - config["margin"][2]
    )

    # person size in first row as tuple
    sizePersonMax = (
        printableLengthPaperY / (config["nPersonsFront"] * config["imgRatio"]),
        printableLengthPaperY / config["nPersonsFront"]
    )

    # person size in last row as tuple
    sizePersonMin = (
        printableLengthPaperY / (config["nPersonsBack"] * config["imgRatio"]),
        printableLengthPaperY / config["nPersonsBack"]
    )

    # calculate imaginery distance of persons in last row
    # using ray projection
    # first row in 1m distance from camera
    alphaStart = np.arctan(sizePersonMax[0] / 2)
    alphaEnd = sizePersonMin[0] / sizePersonMax[0] * alphaStart
    dMax = sizePersonMax[0] / (2 * np.tan(alphaEnd / 2))
    print("max distance in m: " + str(dMax))

    lines = []

    # sum of width of all lines
    totalWidth = 0

    # calculate the size and position for each line
    for i in range(1, config["nLines"] + 1):
        # calculate width and heigth based on imaginery distance
        width = np.arctan(sizePersonMax[0] / 2 / i) / alphaStart * sizePersonMax[0]
        height = width*config["imgRatio"]

        # append new line
        lines.append(
            {
                "height": height,
                "width": width,
                # padding for each line
                # every second line has a padding to create a more natural placement
                "dy": config["paperSize"][1]
                - (i + 1) % 2 * height / 2
                - config["margin"][2]
                - height,
                # number of persons per line
                "nElems": int(
                    (
                        config["paperSize"][1]
                        - ((i + 1) % 2 * height / 2)
                        - config["margin"][0]
                        - config["margin"][2]
                    )
                    / height
                ),
            }
        )
        totalWidth += lines[-1]["width"]

    # scale factor for each line
    scaleY = (
        config["paperSize"][0] - config["margin"][1] - config["margin"][1]
    ) / totalWidth

    # calculate x-position of each axis
    for i, elem in enumerate(lines):
        if i == 0:
            elem["dx"] = config["paperSize"][0] - elem["width"] - config["margin"][1]
        else:
            elem["dx"] = lines[i - 1]["dx"] - elem["width"] * scaleY

    # load or create Mask Image based on config dpi
    cm_to_px = config["dpiMask"] / 2.54
    w_mask = config["paperSize"][0] * cm_to_px
    h_mask = config["paperSize"][1] * cm_to_px
    
    # calculate the center of the paper in px
    center_px = [
        config["paperSize"][0] * 0.5 * cm_to_px,
        config["paperSize"][1] * 0.5 * cm_to_px,
    ]

    # create the final grid
    # the coordinates are in cm
    # position coords are defined from center of the paper
    for line in lines:
        for i in range(line["nElems"]):
            # get absolute coords
            x = line["dx"]
            y = line["dy"] - i * line["height"]

            # calculate pixel coords
            x_px = int(x*cm_to_px)
            y_px = int(y*cm_to_px)
            h_px = int(line["height"]*cm_to_px)
            w_px = int(line["width"]*cm_to_px)

            # center coords to object
            x += 0.5 * line["width"]
            y += 0.5 * line["height"]

            # calculate distance from center
            xcenter = config["paperSize"][0] * 0.5
            ycenter = config["paperSize"][1] * 0.5

            x -= xcenter
            y -= ycenter

            grid.append(
                {"height": line["height"], "width": line["width"], "x": x, "y": y, "xPx": x_px, "yPx": y_px, "widthPx": w_px, "heightPx": h_px}
            ) 

    # load or init mask image
    mask_img = cv2.imread(output_folder + "mask.bmp")
    if (
        mask_img is None
        or mask_img.shape[1] != int(w_mask)
        or mask_img.shape[0] != int(h_mask)
    ):
        mask_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # load or init preview image
    preview_img = cv2.imread(output_folder + "preview.bmp")
    if (
        preview_img is None
        or preview_img.shape[1] != int(w_mask)
        or preview_img.shape[0] != int(h_mask)
    ):
        preview_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)

def start_new_drawing():
    # clean up all files from /out/ into a timestamped subfolder
    target_dir = os.path.join(os.getcwd(),output_folder, datetime.datetime.now().strftime('%y%m%d_%H%M'))
    os.makedirs(target_dir)
    
    file_names = os.listdir(output_folder)
    
    for file_name in file_names:
        if "." in file_name: # move only files
            shutil.move(os.path.join(output_folder+"/", file_name), target_dir)

    # reset the mask and preview by creating a new grid
    grid = []
    create_grid(grid)

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
    canvas_full = True # this does not work if its only set inside this function. we set it before calling this function as well
    
    # (b) sign the document with an pre-made svg signature 
    

# mask image, update image mask and export image to hpgl
def export_image(output_image, mask):
    global run_params
    global mask_img
    global preview_img
    global config
    global hardware_config

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
    cv2.imwrite(output_folder + "preview.bmp", preview_img)
    cv2.imwrite(output_folder + "mask.bmp", mask_img)
    cv2.imwrite(output_folder + str(currentId) + ".bmp", output_image)

    # run vpype scripts to trace image, position on paper, export as hpgl and plot
    ser.write('h'.encode())
    if run_params["scripts"]:
        # trace
        bashCommand = "sh 0_trace_image.sh " + output_folder + str(currentId)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # place vector image on paper
        bashCommand = (
            "sh 1_scale_position.sh " + output_folder + str(currentId) + " "
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
            bashCommand = "sh 4_mergeLayers.sh " + output_folder + str(currentId) + "_scaled"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        # export as hpgl
        bashCommand = "sh 3_export_hpgl.sh " + output_folder + str(currentId) + "_scaled"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # print hpgl
    if run_params["print"]:  
        bashCommand = "python hp7475a_send.py " + output_folder + str(currentId) + "_scaled.hpgl -p " + hardware_config["plotterSerial"]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # update id and write to file
    cf["currentId"] += 1
    with open("counter.json", "w") as f:
        json.dump(cf, f)

def export_test_grid():

    for count, elemSettings in enumerate(grid):

        print("exporting " + str(count+1) + " / " + str(len(grid)))

    # place vector image on paper
        bashCommand = (
            "sh 5_create_test_grid.sh " + output_folder + "singleGrid "
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
cv2.createTrackbar("line_size", "output", 2, 7, set_line_size)
cv2.createTrackbar("blur_edge", "output", 2, 7, set_blur_edge)
cv2.createTrackbar("blur_image", "cam", 10, 40, set_blur_image)
cv2.createTrackbar("circle_border", "cam", 22, 80, set_circle_border)
cv2.createTrackbar("circle_radius", "cam", 10, 30, set_circle_radius)
cv2.createTrackbar("clip_limit", "cam", 1, 30, set_clip_limit)

# create grid
create_grid(grid)

#show usage/documentation in console
show_usage()

# select capture device
cap = cv2.VideoCapture(hardware_config["camId"])

# capture loop
while cap.isOpened():
    # count fps
    start_time = time.time() # start time of the loop

    # read image
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rotate image since camera is rotated as well
    if cam_img_rotate != 0:
        image = cv2.rotate(image, img_rotater[cam_img_rotate])

    # init output image with zeros
    output_image = np.zeros(
        (config["croppedRes"], int(config["croppedRes"] * config["imgRatio"]), 3),
        np.uint8,
    )

    # calulate image ratio and resize image to config settings
    ratio = image.shape[0] / image.shape[1]

    # landscape
    # image = cv2.resize(image, (int(config["resImg"] / ratio), config["resImg"]))
    
    # portrait
    image = cv2.resize(image, (config["resImg"], int(config["resImg"] * ratio)))

    # set not writable for more performance
    image.flags.writeable = False

    # detect faces
    faces = face_detection.process(image)

    # continue if face detected
    if faces.detections:
        if not is_face_detected:
            time_last_detection = time.time()

        is_face_detected = True
        bb = faces.detections[0].location_data.relative_bounding_box
        
        # get segmented image
        segmentation_bg = selfie_segmentation.process(image)

        # prepare and cut out img
        condition = np.stack((segmentation_bg.segmentation_mask,) * 3, axis=-1) > 0.1
        # init bg image
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (255, 255, 255) 

        # equalize histogram for better contrasts
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur part around face
        gray_blur = cv2.medianBlur(gray, blur_image)
        circular_mask = np.zeros(gray.shape, dtype=np.uint8)
        c_x = int((bb.xmin + 0.5*bb.width)*gray.shape[1])
        c_y = int((bb.ymin + 0.5*bb.height)*gray.shape[0])
        r = int(max(bb.height*gray.shape[0], bb.width*gray.shape[1])*0.5*circle_radius)
        cv2.circle(circular_mask, (c_x, c_y), r, (255), thickness=-1)
        circular_mask = cv2.GaussianBlur(circular_mask, (circle_border, circle_border), 0)

        gray = np.uint8(gray * (circular_mask / 255) + gray_blur * (1 - (circular_mask / 255)))

        # we use clahe filter that creates better results than simple historgram equalize (see below)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        cv2.imshow("cam2", gray)

        # detect edges
        edge = edge_mask(gray, line_size, blur_edge)
        #edge = pencil_sketch(gray, line_size, blur_edge)

        # remove bg from edge image
        out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cut_image = np.where(condition, out, bg_image)

        # cut image to final size
        d_x = cut_image.shape[0] - output_image.shape[0]
        d_x *= 0.5
        d_x = int(d_x)
        d_y = cut_image.shape[1] - output_image.shape[1]
        d_y *= 0.5
        d_y = int(d_y)

        output_image = cut_image[
            d_x: cut_image.shape[0] - d_x, d_y: cut_image.shape[1] - d_y
        ]

        # create a mask to dstinct back and foreground
        mask = np.where(condition, image, bg_image)
        mask = mask[d_x: cut_image.shape[0] - d_x, d_y: cut_image.shape[1] - d_y]
        gray2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray2, 254, 255, cv2.THRESH_BINARY)

        # show output image
        cv2.imshow("output", output_image)
    else:
        is_face_detected = False


    # autodraw
    if autodraw and is_face_detected and time.time() - time_last_detection > time_delay_photo:
        try:
            export_image(output_image, mask)
        except:
            canvas_full = True # prevent further drawing till [s] is pressed
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
            canvas_full = True # prevent further drawing till [s] is pressed
            page_is_full()
    # if key was 'r' then rotate image
    elif k == ord("r"):
        cam_img_rotate=(cam_img_rotate+1)%4 ## select next rotation style (none, 90ccw,180,90cw)
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
    elif k == ord("g"):
        print("I guess I'll export only a grid for now.")
        export_test_grid()
    elif k == ord("a"):
        print("I guess I'll export only a grid for now.")
        export_test_grid()
    elif k & 0xFF == 27:
        break

    if (show_fps):
        print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

"""

# todo autoprint if person sits for some seconds

    image.flags.writeable = False
    croppedImg.flags.writeable = False
    results = selfie_segmentation.process(croppedImg)

    image.flags.writeable = True
    croppedImg.flags.writeable = True
    croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    if bg_image is None:
      bg_image = np.zeros(croppedImg.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR
    #output_image = np.where(condition, image, bg_image)

     ## contrast stretch
    #xp = [0, 64, 128, 192, 255]
    #fp = [0, 16, 128, 240, 255]
    #x = np.arange(256)
    #table = np.interp(x, xp, fp).astype('uint8')
    #image = cv2.LUT(image, table)

    gray = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    
    gray = cv2.equalizeHist(gray)
    edge = edge_mask(gray,11,5)
    out = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR);
    output_image = np.where(condition, out, bg_image)
    mask = np.where(condition, croppedImg, bg_image)
    gray2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray2,254,255,cv2.THRESH_BINARY)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
"""


cap.release()
cv2.destroyAllWindows()
