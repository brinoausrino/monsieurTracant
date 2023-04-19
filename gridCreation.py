import numpy as np
import cv2


def create_grid(config):
    grid = []

    width = config["layout"]["paperSize"][0] - config["layout"]["margin"][1] - config["layout"]["margin"][3]
    height = config["layout"]["paperSize"][1] - config["layout"]["margin"][0] - config["layout"]["margin"][2]
    
    w_img = config["grid"]["size_single_img"][0]
    h_img = config["grid"]["size_single_img"][1]

    # load or create Mask Image based on config dpi
    cm_to_px = config["layout"]["dpi"] / 2.54

    # calculate the center of the paper in px
    center_px = [
        config["layout"]["paperSize"][0] * 0.5 * cm_to_px,
        config["layout"]["paperSize"][1] * 0.5 * cm_to_px,
    ]

    # center coords to object
    cx = 0.5 * config["layout"]["paperSize"][0]
    cy = 0.5 * config["layout"]["paperSize"][1]

    # calculate number of images
    xn_images = int(width / (w_img + config["grid"]["padding"]))
    yn_images = int(height / (h_img + config["grid"]["padding"]))

    # center images
    dy = config["layout"]["margin"][0] + (width - xn_images*w_img)*0.5 -cy
    dx = config["layout"]["margin"][1] + (height - h_img)*0.5 -cx

    for y_t in range(yn_images):
        for x_t in range(xn_images):
            
            # calculate pixel coords
            x_px = int(dx*cm_to_px)
            y_px = int(dy*cm_to_px)
            h_px = int(h_img*cm_to_px)
            w_px = int(w_img*cm_to_px)
    
            grid.append(
                {
                    "height": h_img, 
                    "width": w_img, 
                    "x": dx, 
                    "y": dy, 
                    "xPx": x_px, 
                    "yPx": y_px, 
                    "widthPx": w_px, 
                    "heightPx": h_px
                }
            ) 
            dx += w_img + config["grid"]["padding"]
        dy += h_img + config["grid"]["padding"]

    return {
        "cm_to_px": cm_to_px,
        "center_px": center_px,
        "grid": grid
    }

def create_single(config):
    grid = []

    width = config["layout"]["paperSize"][0] - config["layout"]["margin"][1] - config["layout"]["margin"][3]
    height = config["layout"]["paperSize"][1] - config["layout"]["margin"][0] - config["layout"]["margin"][2]
    dy = config["layout"]["margin"][0]
    dx = config["layout"]["margin"][1]

    # load or create Mask Image based on config dpi
    cm_to_px = config["layout"]["dpi"] / 2.54
    w_mask = config["layout"]["paperSize"][0] * cm_to_px
    h_mask = config["layout"]["paperSize"][1] * cm_to_px
    
    # calculate the center of the paper in px
    center_px = [
        config["layout"]["paperSize"][0] * 0.5 * cm_to_px,
        config["layout"]["paperSize"][1] * 0.5 * cm_to_px,
    ]

    # calculate pixel coords
    x_px = int(dx*cm_to_px)
    y_px = int(dy*cm_to_px)
    h_px = int(height*cm_to_px)
    w_px = int(width*cm_to_px)

    # center coords to object
    x = 0.5 * config["layout"]["paperSize"][0]
    y = 0.5 * config["layout"]["paperSize"][1]

    grid.append(
        {
            "height": height, 
            "width": width, 
            "x": dx, 
            "y": dy, 
            "xPx": x_px, 
            "yPx": y_px, 
            "widthPx": w_px, 
            "heightPx": h_px
        }
    ) 

    # load or init mask image
    mask_img = cv2.imread(config["outputFolder"] + "mask.bmp")
    if (
        mask_img is None
        or mask_img.shape[1] != int(w_mask)
        or mask_img.shape[0] != int(h_mask)
    ):
        mask_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # load or init preview image
    preview_img = cv2.imread(config["outputFolder"] + "preview.bmp")
    if (
        preview_img is None
        or preview_img.shape[1] != int(w_mask)
        or preview_img.shape[0] != int(h_mask)
    ):
        preview_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)

    return {
        "cm_to_px": cm_to_px,
        "mask_img": mask_img,
        "preview_img": preview_img,
        "center_px": center_px,
        "grid": grid
    }

# creates person grid based on config file
# some calculations me be confuding, since the final image is rotated 90°
# therefore x and y are swapped
def create_perspective_grid(config):
    grid = []

    # printable length on paper 
    printableLengthPaperY = (
        config["layout"]["paperSize"][1] - config["layout"]["margin"][0] - config["layout"]["margin"][2]
    )

    # person size in first row as tuple
    sizePersonMax = (
        printableLengthPaperY / (config["grid"]["nPersonsFront"] * config["grid"]["imgRatio"]),
        printableLengthPaperY / config["grid"]["nPersonsFront"]
    )

    # person size in last row as tuple
    sizePersonMin = (
        printableLengthPaperY / (config["grid"]["nPersonsBack"] * config["grid"]["imgRatio"]),
        printableLengthPaperY / config["grid"]["nPersonsBack"]
    )

    # calculate imaginery distance of persons in last row
    # using ray projection
    # first row in 1m distance from camera
    alphaStart = np.arctan(sizePersonMax[0] / 2)
    alphaEnd = sizePersonMin[0] / sizePersonMax[0] * alphaStart
    dMax = sizePersonMax[0] / (2 * np.tan(alphaEnd / 2))

    lines = []

    # sum of width of all lines
    totalWidth = 0

    # calculate the size and position for each line
    for i in range(1, config["grid"]["nLines"] + 1):
        # calculate width and heigth based on imaginery distance
        width = np.arctan(sizePersonMax[0] / 2 / i) / alphaStart * sizePersonMax[0]
        height = width*config["grid"]["imgRatio"]

        # append new line
        lines.append(
            {
                "height": height,
                "width": width,
                # padding for each line
                # every second line has a padding to create a more natural placement
                "dy": config["layout"]["paperSize"][1]
                - (i + 1) % 2 * height / 2
                - config["layout"]["margin"][2]
                - height,
                # number of persons per line
                "nElems": int(
                    (
                        config["layout"]["paperSize"][1]
                        - ((i + 1) % 2 * height / 2)
                        - config["layout"]["margin"][0]
                        - config["layout"]["margin"][2]
                    )
                    / height
                ),
            }
        )
        totalWidth += lines[-1]["width"]

    # scale factor for each line
    scaleY = (
        config["layout"]["paperSize"][0] - config["layout"]["margin"][1] - config["layout"]["margin"][1]
    ) / totalWidth

    # calculate x-position of each axis
    for i, elem in enumerate(lines):
        if i == 0:
            elem["dx"] = config["layout"]["paperSize"][0] - elem["width"] - config["layout"]["margin"][1]
        else:
            elem["dx"] = lines[i - 1]["dx"] - elem["width"] * scaleY

    # load or create Mask Image based on config dpi
    cm_to_px = config["layout"]["dpi"] / 2.54
    w_mask = config["layout"]["paperSize"][0] * cm_to_px
    h_mask = config["layout"]["paperSize"][1] * cm_to_px
    
    # calculate the center of the paper in px
    center_px = [
        config["layout"]["paperSize"][0] * 0.5 * cm_to_px,
        config["layout"]["paperSize"][1] * 0.5 * cm_to_px,
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
            xcenter = config["layout"]["paperSize"][0] * 0.5
            ycenter = config["layout"]["paperSize"][1] * 0.5

            x -= xcenter
            y -= ycenter

            grid.append(
                {"height": line["height"], "width": line["width"], "x": x, "y": y, "xPx": x_px, "yPx": y_px, "widthPx": w_px, "heightPx": h_px}
            ) 

    # load or init mask image
    mask_img = cv2.imread(config["outputFolder"] + "mask.bmp")
    if (
        mask_img is None
        or mask_img.shape[1] != int(w_mask)
        or mask_img.shape[0] != int(h_mask)
    ):
        mask_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    
    # load or init preview image
    preview_img = cv2.imread(config["outputFolder"] + "preview.bmp")
    if (
        preview_img is None
        or preview_img.shape[1] != int(w_mask)
        or preview_img.shape[0] != int(h_mask)
    ):
        preview_img = np.ones((int(h_mask), int(w_mask)), np.uint8) * 255
    else:
        preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)

    return {
        "cm_to_px" : cm_to_px,
        "mask_img" : mask_img,
        "preview_img" : preview_img,
        "center_px":center_px,
        "grid":grid
    }