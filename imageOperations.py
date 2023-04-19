import cv2
from informativeDrawings import informativeDrawings
from traceSkeleton import trace_skeleton
import numpy as np
from scipy import ndimage
import math
import pyfastnoisesimd as fns

inf_drawings = {}
noisefield = None
vignette_mask = None


def img_to_polys(img,settings={"threshold":{"min":128,
                                  "max":255},
                 "vignette":{"layer":128},
                 "hatching":{"active":False},
                 "pretreating" : {"active":False,
                                  "clipLimit_clahe":2}},id="img to polys",mask_img=None):

    # pretreat
    if settings["pretreating"]["active"]:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = img_yuv[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=settings["pretreating"]["clipLimit_clahe"], tileGridSize=(8, 8))
        img = clahe.apply(img)
    else:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = img_yuv[:,:,0]
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if settings["debug"]:
            cv2.imshow(id + ":pretreat", img)
    # hatching
    if settings["hatching"]["active"]:
        img = cv2.bitwise_not(img)
        if settings["hatching"]["type"] == "noise":
            pattern = create_noise_pattern(img, settings["hatching"]["d_lines"], settings["hatching"]["angle_lines"], settings["hatching"]["w_lines"], 
                                            offset=settings["hatching"]["offset"])
        elif settings["hatching"]["type"] == "crossNoise":
            pattern = create_cross_noise_pattern(img, settings["hatching"]["d_lines"], settings["hatching"]["angle_lines"], settings["hatching"]["w_lines"], 
                                            offset=settings["hatching"]["offset"])
        else:
            pattern = create_hatch_pattern(img,settings["hatching"]["d_lines"], settings["hatching"]["angle_lines"], settings["hatching"]["w_lines"])

        # mask image with pattern
        img = cv2.bitwise_and(img, img, mask=pattern)
        img = cv2.bitwise_not(img)
        
        #img = cv2.bitwise_not(img)
        if settings["debug"]:
            cv2.imshow(id + ":hatching", img)
    
    # masking
    if not mask_img is None:
        img = cv2.bitwise_and(img,mask_img)
        img = cv2.bitwise_not(img)
        
        if settings["debug"]:
            cv2.imshow(id + ":masking", img)
    
    # threshold and vignette
    img = clamp_threshold(img, settings["threshold"]["min"],settings["threshold"]["max"])
    if settings["debug"]:
            cv2.imshow(id + ":threshold", img)
    
    img = add_vignette(img,settings["vignette"]["layer"],settings["background"])
    
    if settings["debug"]:
            cv2.imshow(id + ":vignette", img)
    
    img = cv2.bitwise_not(img)
    
    if settings["debug"]:
        cv2.imshow(id + ":tracing", img)
    
    return trace_image(img,False,settings["traceMinLength"])



def extract_contours(img, render_quality=768, export_preview=False, tracing_resolution = 2):
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
    preview_img = cv2.resize(preview_img, (img.shape[1], img.shape[0]))
    preview_img = add_vignette(preview_img,511,"white")
    preview_img = cv2.bitwise_not(preview_img)
    #p2 = preview_img.copy()

    _, preview_img = cv2.threshold(preview_img, 128, 255, cv2.THRESH_BINARY)

    p1 = trace_image(preview_img, False,tracing_resolution)
    #cv2.imshow("line "+str(tracing_resolution), i1)
    
    # hatch line tracing
    #pattern = create_hatch_pattern(p2,3, 45, 1)
   # cv2.imshow("l", p2)
    #p2 = cv2.bitwise_and(p2, p2, mask=pattern)
    #_, p2 = cv2.threshold(p2, 128, 255, cv2.THRESH_BINARY)
   # cv2.imshow("linHatch", p2)
    #px = trace_image(p2, False,2)
    
    return p1 #px +p1
    #return trace_image(preview_img, export_preview)

def add_vignette(img,threshold=128,bg = "black"):
    global vignette_mask
    if vignette_mask is None:
        vignette_mask = create_noise(frequency=0.015, octaves=4, lacunarity=2.1, gain=0.45)
    n = ((vignette_mask[threshold%512]+0.5)*255).astype(np.uint8)
    n = cv2.resize(n, (img.shape[1],img.shape[0]))
    circular_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    cv2.circle(circular_mask, (int(img.shape[1]*0.5),int(img.shape[0]*0.5)), int(img.shape[0]*0.5), (255,255,255), thickness=-1)
    n = cv2.bitwise_and(n, circular_mask)
    circular_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    cv2.circle(circular_mask, (int(img.shape[1]*0.5),int(img.shape[0]*0.5)), int(img.shape[0]*0.43), (255,255,255), thickness=-1)
    n = cv2.bitwise_or(n, circular_mask)
    _, n = cv2.threshold(n, 128, 255, cv2.THRESH_BINARY)
    if bg == 'black':
        img = cv2.bitwise_and(n,img)
    else:
        img = cv2.bitwise_not(img)
        img = cv2.bitwise_and(n,img)
        img = cv2.bitwise_not(img)
    #n = cv2.bitwise_not(n)
    if img.ndim == 2:
        return img #cv2.add(n,img)
    else:
        return cv2.add(np.dstack((n,n,n)),img)


def hatch_area(img, threshold=128, angle_lines=45, d_lines=20, w_lines=3, invert_image=False, export_preview=False, offset=0, threshold_max = 255):
    # load and threshold image
    preview_img = None
    if img.ndim == 2:
        preview_img = img
    else:
        preview_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if not invert_image:
        preview_img = cv2.bitwise_not(preview_img)

    # pattern = create_hatch_pattern(preview_img,d_lines,angle_lines,w_lines)
    pattern = create_noise_pattern(preview_img, d_lines, angle_lines, w_lines, offset=offset)

    # mask image with pattern
    preview_img = cv2.bitwise_and(preview_img, preview_img, mask=pattern)
    preview_img = clamp_threshold(preview_img,threshold,threshold_max)

    # trace image
    return trace_image(preview_img, export_preview)

def clamp_threshold(img, threshold_min,threshold_max):
    img_temp = None
    if threshold_max == 255:
        _, img_temp = cv2.threshold(img, threshold_min, 255, cv2.THRESH_BINARY)
    else:
        img_temp = img.copy()
        for y in range(0, img_temp.shape[0]):
            for x in range(0, img_temp.shape[1]):
                img_temp[y, x] = 0 if img_temp[y, x] >= threshold_min and img_temp[y, x] <= threshold_max else 255
    return img_temp

def create_hatch_pattern(preview_img, d_lines, angle_lines, w_lines):
    w_pattern = int(math.sqrt(max(preview_img.shape[0], preview_img.shape[1]) ** 2 / 2)*2)
    pattern = np.zeros((w_pattern, w_pattern, 1), np.uint8,)
    for y in range(0, pattern.shape[0], d_lines):
        cv2.line(pattern, (y, 0), (y, pattern.shape[0]), 255, w_lines)

    pattern = ndimage.rotate(pattern, angle_lines, reshape=False, order=0, prefilter=False)
    pattern = pattern[
        int(0.5*(pattern.shape[0] - preview_img.shape[0])): int(pattern.shape[0] - 0.5*(pattern.shape[0] - preview_img.shape[0])),
        int(0.5*(pattern.shape[1] - preview_img.shape[1])): int(pattern.shape[1] - 0.5*(pattern.shape[1] - preview_img.shape[1]))
    ]
    return pattern

def create_sine_pattern(preview_img, d_lines, angle_lines, w_lines):
    res = 1
    h_wave = 7
    w_wave = 15

    w_wave /= math.pi

    w_pattern = int(math.sqrt(max(preview_img.shape[0], preview_img.shape[1]) ** 2 / 2)*2)
    pattern = np.zeros((w_pattern, w_pattern, 1), np.uint8,)
    for y in range(0, pattern.shape[0], d_lines):
        for x in range(0, pattern.shape[1], res):
            p1y = int(y + math.sin(x/w_wave)*h_wave)
            p2y = int(y + math.sin((x+res)/w_wave)*h_wave)
            cv2.line(pattern, (p1y, x), (p2y, x+res), 255, w_lines)
            # cv2.line(pattern,(y + math.sin(x/w_wave)*h_wave, x),(y + math.sin((x+res)/w_wave)*h_wave,x+res),255,w_lines)

    pattern = ndimage.rotate(pattern, angle_lines, reshape=False, order=0, prefilter=False)
    pattern = pattern[
        int(0.5*(pattern.shape[0] - preview_img.shape[0])): int(pattern.shape[0] - 0.5*(pattern.shape[0] - preview_img.shape[0])),
        int(0.5*(pattern.shape[1] - preview_img.shape[1])): int(pattern.shape[1] - 0.5*(pattern.shape[1] - preview_img.shape[1]))
    ]
    return pattern

def create_noise_pattern(preview_img, d_lines, angle_lines, w_lines, scale=20,offset = 0,frequency = 0.05,octaves =4,lacunarity = 2.1):
    global noisefield
    #if noisefield is None:
    noisefield = create_noise(frequency=frequency,octaves=octaves,lacunarity=lacunarity)
    res = 3

    w_pattern = int(math.sqrt(max(preview_img.shape[0], preview_img.shape[1]) ** 2 / 2)*2)
    pattern = np.zeros((w_pattern, w_pattern, 1), np.uint8,)

    noise_scale = 512/w_pattern

    for y in range(0, pattern.shape[0], d_lines):
        for x in range(0, pattern.shape[1]-res, res):
            p1y = int(y + noisefield[int(y*noise_scale), int(x*noise_scale),int(angle_lines + offset)%512]*scale)
            p2y = int(y + noisefield[int(y*noise_scale), int((x+res)*noise_scale),int(angle_lines+ offset)%512]*scale)
            cv2.line(pattern, (p1y, x), (p2y, x+res), 255, w_lines)

    pattern = ndimage.rotate(pattern, angle_lines, reshape=False, order=0, prefilter=False)
    pattern = pattern[
        int(0.5*(pattern.shape[0] - preview_img.shape[0])): int(pattern.shape[0] - 0.5*(pattern.shape[0] - preview_img.shape[0])),
        int(0.5*(pattern.shape[1] - preview_img.shape[1])): int(pattern.shape[1] - 0.5*(pattern.shape[1] - preview_img.shape[1]))
    ]
    return pattern

def create_cross_noise_pattern(preview_img, d_lines, angle_lines, w_lines, scale=20,offset = 0,frequency = 0.05,octaves =4,lacunarity = 2.1):
    base = create_noise_pattern(preview_img, d_lines, 
                                                  angle_lines,w_lines ,
                                                  scale=scale,
                                                  frequency=frequency,
                                                  octaves=octaves,
                                                  lacunarity=lacunarity)

    cross = create_noise_pattern(preview_img, d_lines*2, (angle_lines+90)%360, 
                                                 w_lines,offset = 230,
                                                 scale=scale*2,
                                                 frequency=frequency,
                                                 octaves=octaves,
                                                lacunarity=lacunarity)
    cross = cv2.bitwise_not(cross)
    pattern = cv2.bitwise_and(base,cross)
    return pattern

def create_noise(frequency=0.05, octaves=4, lacunarity=2.1, gain=0.45):
    shape = [512, 512, 512]
    seed = np.random.randint(2**31)
    N_threads = 4
    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    perlin.frequency = frequency
    perlin.noiseType = fns.NoiseType.Perlin
    perlin.fractal.octaves = octaves
    perlin.fractal.lacunarity = lacunarity
    perlin.fractal.gain = gain
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    return perlin.genAsGrid(shape)

def trace_image(img, export_preview,csize=10):
    polys = trace_skeleton.from_numpy(img,csize)
    if export_preview:
        img[:] = 255
        img = create_preview_from_polys(polys, img)
        return (polys, img)
    return polys

def create_preview_from_polys(polys, img, color=(0, 0, 0), line_thickness=1):
    for l in polys:
        for i in range(0, len(l)-1):
            cv2.line(img, (l[i][0], l[i][1]), (l[i+1][0], l[i+1][1]), color, line_thickness)
    return img

def bgr_to_cmyk(img):
    # https://stackoverflow.com/questions/60814081/how-to-convert-a-rgb-image-into-a-cmyk
    
    bgr = img.astype(float)/255.
    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1-bgr[...,2] - K)/(1-K)
        M = (1-bgr[...,1] - K)/(1-K)
        Y = (1-bgr[...,0] - K)/(1-K)
    # Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
    Y, M, C, K = cv2.split(CMYK)
    return (C,M,Y,K)

def bgr_to_cmy(img):
    bgrdash = img.astype(np.float)/255.
    
    # Calculate C
    C = 1-bgrdash[...,2]

        # Calculate M
    M = 1-bgrdash[...,1]

        # Calculate Y
    Y = 1-bgrdash[...,0]
        
    C = (C*255).astype(np.uint8)
    M = (M*255).astype(np.uint8)
    Y = (Y*255).astype(np.uint8)
    
    return (C,M,Y)