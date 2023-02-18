import cv2
from informativeDrawings import informativeDrawings
from traceSkeleton import trace_skeleton
import numpy as np
from scipy import ndimage
import math

inf_drawings = {}


def extract_contours(img,render_quality = 768, export_preview = False):
    global inf_drawings

    if not inf_drawings:
        # init info drawings
        inf_drawings = informativeDrawings.Options()
        inf_drawings.dataroot = "inf_drawings_t.png"
        informativeDrawings.init_nn(inf_drawings)

    # save temp
    cv2.imwrite("inf_drawings_t.png", img)

    # proceed neural network
    inf_drawings.size = render_quality
    informativeDrawings.proceed_conversion(inf_drawings)

    # load proceeded image
    preview_img = cv2.imread("informativeDrawings/results/contour_style/inf_drawings_t_out.png", 0)
    preview_img = cv2.resize(preview_img, (img.shape[1],img.shape[0]))
    preview_img = cv2.bitwise_not(preview_img)

    _,preview_img = cv2.threshold(preview_img, 128, 255, cv2.THRESH_BINARY)

        #open('output_'+str(q)+'.svg','w').write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{preview_img.shape[1]}" height="{preview_img.shape[0]}"><path stroke="red" fill="none" d="'+" ".join(["M"+" ".join([f'{x[0]},{x[1]}' for x in y]) for y in polys])+'"/></svg>')

    return trace_image(preview_img, export_preview)



def hatch_area(img, threshold = 128, angle_lines = 45, d_lines = 20, w_lines=3, invert_image = False, export_preview = False):
    # load and threshold image
    preview_img = None
    if img.ndim == 2:
        preview_img = img
    else:
        preview_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if not invert_image:
        preview_img = cv2.bitwise_not(preview_img)
    _,preview_img = cv2.threshold(preview_img,threshold,255,cv2.THRESH_BINARY)
    
    # create hatch pattern
    w_pattern = int(math.sqrt(max(preview_img.shape[0],preview_img.shape[1]) ** 2 / 2)*2)
    pattern = np.zeros((w_pattern,w_pattern, 1),np.uint8,)
    for y in range(0,pattern.shape[0],d_lines):
        cv2.line(pattern,(y,0),(y,pattern.shape[0]),255,w_lines)

    pattern = ndimage.rotate(pattern, angle_lines,reshape=False,order=0,prefilter=False)
    pattern = pattern[
        int(0.5*(pattern.shape[0] - preview_img.shape[0])): int(pattern.shape[0] - 0.5*(pattern.shape[0] - preview_img.shape[0])),
        int(0.5*(pattern.shape[1] - preview_img.shape[1])): int(pattern.shape[1] - 0.5*(pattern.shape[1] - preview_img.shape[1]))
    ]

    # mask image with pattern
    preview_img = cv2.bitwise_and(preview_img, preview_img, mask=pattern)

    # trace image
    return trace_image(preview_img, export_preview)


def trace_image(img, export_preview):
    polys = trace_skeleton.from_numpy(img)
    if export_preview:
        img[:] = 255
        img = create_preview_from_polys(polys, img)
        return(polys,img)
    return polys

def create_preview_from_polys(polys,img, color =(0,0,0)):
    for l in polys:
        for i in range(0,len(l)-1):
            cv2.line(img,(l[i][0],l[i][1]),(l[i+1][0],l[i+1][1]),color)
    return img