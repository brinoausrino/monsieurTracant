# imports
import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import json

##################
# global variables
##################

# person detection and automatic drawing
time_last_detection = time.time()
time_delay_photo = 5

# script options
run_params = {
    "scripts": True, 
    "print": False,
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

# calculation and config params
cm_to_px = 1
center_px = [0, 0]
config = {}

# filter variables
line_size = 11
blur_edge = 5
blur_image = 8

#########
# functions
#########

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
        #height = np.arctan(sizePersonMax[1] / 2 / i) / alphaStart * sizePersonMax[1]
        # width = np.arctan(sizePersonMax[0] / 2 / i) / alphaStart * sizePersonMax[0]
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
    cv2.imwrite(output_folder + "preview.bmp", preview_img)
    cv2.imwrite(output_folder + "mask.bmp", mask_img)
    cv2.imwrite(output_folder + str(currentId) + ".bmp", output_image)

    # run vpype scripts to trace image, position on paper, export as hpgl and plot
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
        bashCommand = "python hp7475a_send.py " + output_folder + str(currentId) + "_scaled.hpgl -p /dev/ttyUSB0"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # update id and write to file
    cf["currentId"] += 1
    with open("counter.json", "w") as f:
        json.dump(cf, f)


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
cv2.createTrackbar("line_size", "output", 2, 7, set_line_size)
cv2.createTrackbar("blur_edge", "output", 2, 7, set_blur_edge)
cv2.createTrackbar("blur_image", "output", 2, 20, set_blur_image)

# create grid
create_grid(grid)

# select capture device
cap = cv2.VideoCapture(2)

# capture loop
while cap.isOpened():

    # read image
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # rotate image since camera is roted as well
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
    is_detected = False
    if faces.detections:
        is_detected = True
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
        
        # we use clahe filter that creates better results than simple historgram equalize (see below)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # blur part around face
        # todo : create circular mask with soft edges for soft blending
        gray_blur = cv2.medianBlur(gray, blur_image)
        
        gray_blur.flags.writeable = True
        bb_xmin = int(bb.xmin*gray_blur.shape[1])
        bb_xmax = int((bb.xmin+bb.width)*gray_blur.shape[1])
        bb_ymin = int(bb.ymin*gray_blur.shape[0])
        bb_ymax = int((bb.ymin+bb.height)*gray_blur.shape[0])
        gray_blur[bb_ymin: bb_ymax, bb_xmin: bb_xmax]= gray[bb_ymin: bb_ymax, bb_xmin: bb_xmax]

        gray = clahe.apply(gray_blur)
        cv2.imshow("gray", gray)

        # detect edges
        edge = edge_mask(gray, line_size, blur_edge)

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

    # click 'p' and start exporting and plotting
    if cv2.waitKey(33) == ord("p") and is_detected:
        export_image(output_image, mask)

    elif cv2.waitKey(5) & 0xFF == 27:
        break

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
