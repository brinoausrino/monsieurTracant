# imports
import time
import json
import datetime

import cv2
import numpy as np
import mediapipe as mp

import gridCreation
import imageOperations
import exportOperations
from comfyAdapter import comfyAdapter

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
bg_image = {}

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
circle_radius = 1.2
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

	erodes = [0,2,3,2]
	dilates = [0,2,3,2]
	#dilates = [0,2,2,2]
	#blurs = [0,3,13,23]


	scale = 0.5
	size = (int(image_container[3].shape[1]*scale) , int(image_container[3].shape[0]*scale))

	for i in range(len(image_container)):
		if i >= 2:
			#image_container[i] = cv2.GaussianBlur(image_container[i], (blurs[i],blurs[i]), 0)
			erodeK = np.ones((erodes[i],erodes[i]), np.uint8)
			dilateK =  np.ones((dilates[i],dilates[i]), np.uint8)
			eroded_image = cv2.dilate(image_container[i], erodeK, iterations=1)
			#if i==3:
			#	eroded_image = cv2.erode(eroded_image, dilateK, iterations=1)
			_, binary_thresh = cv2.threshold(eroded_image, 220, 255, cv2.THRESH_BINARY)
			binary_thresh = cv2.bitwise_not(binary_thresh)
			#cv2.imwrite("temp"+str(i)+".png", binary_thresh)
			image_container[i] = binary_thresh.copy()
			image_container[i] = cv2.resize(image_container[i], size)
			line_container.append(imageOperations.trace_image(image_container[i], False,2)) #imageOperations.extract_contours(image_container[i]))
			image_container[i][:] = 255
			image_container[i] = imageOperations.create_preview_from_polys(line_container[i], image_container[i],(0,0,0),1)
		else:
			dilateK =  np.ones((5,5), np.uint8)
			dilated_image = cv2.dilate(image_container[i], dilateK, iterations=1)
			_, binary_thresh = cv2.threshold(dilated_image, 220, 255, cv2.THRESH_BINARY)
			binary_thresh = cv2.bitwise_not(binary_thresh)
			image_container[i] = binary_thresh.copy()
			image_container[i] = cv2.resize(image_container[i], size,cv2.INTER_NEAREST_EXACT)
			line_container.append(imageOperations.trace_image(image_container[i], False,2)) #imageOperations.extract_contours(image_container[i]))
			image_container[i][:] = 255
			image_container[i] = imageOperations.create_preview_from_polys(line_container[i], image_container[i],(0,0,0),1)
	preview = np.concatenate((image_container), axis=1)
	cv2.putText(preview,"Bild auswaehlen- (n abbruch)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0)
	for i in range(len(image_container)):
		cv2.putText(preview,str(i), (20 + i*image_container[0].shape[1],100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
	cv2.imshow("tracant", preview)

def finalize_image(output_image,lines):

	image = cv2.imread("output/image_217.png", cv2.IMREAD_UNCHANGED)  
	image = rgba_to_grayscale_white_background(image)
	image = cv2.resize(image, (output_image.shape[1],output_image.shape[0]))

	f = open("layer.json")
	s = json.load(f)

	# export
	dx = 0
	dy = 0
	wp = 105
	hp = 148

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

	preview = output_image.copy()
	

	preview[:] = 255
		
	for i,item in enumerate(s["hatchWedding"]):
		h = imageOperations.img_to_polys(image,item)
		exportOperations.export_polys_as_hpgl(h, (int(output_image2.shape[1]),int(output_image2.shape[0])), timestamp +str(i) + "_"+ item["color"], wp, hp,"A6_7475A",90,dx,dy)
		preview = imageOperations.create_preview_from_polys(h, preview,(0,0,0),1)
		
	exportOperations.export_polys_as_hpgl(lines, (int(output_image2.shape[1]),int(output_image2.shape[0])), timestamp + "4_black", wp, hp,"A6_7475A",90,dx,dy,True)
	preview = imageOperations.create_preview_from_polys(lines, preview,(0,0,0),1)
	cv2.putText(preview,"Leertaste fuer neue Aufnahme", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0)
	cv2.imshow("tracant", preview)

def finalize_image_group(output_image,lines):

	# get current position in grid from counter.json
	f = open("counter.json")
	cf = json.load(f)
	currentId = cf["currentId"]

	# get size and position settings for current element
	elemSettings = layout["grid"][currentId]

	# rotate images
	#output_image = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	# positioning
	dx = elemSettings["x"]*10
	dy = elemSettings["y"]*10
	wp = elemSettings["width"]*10
	hp = elemSettings["height"]*10

	f = open("layer.json")
	s = json.load(f)

	timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")

	preview = output_image.copy()
	preview = cv2.rotate(preview, cv2.ROTATE_90_COUNTERCLOCKWISE)

	preview[:] = 255
		
	for i,item in enumerate(s["hatchWedding"]):
		h = imageOperations.img_to_polys(output_image,item)
		exportOperations.export_polys_as_hpgl(h, (int(output_image.shape[1]),int(output_image.shape[0])), timestamp +str(i) + "_"+ item["color"], wp, hp,"70_90",270,dx,dy)
		preview = imageOperations.create_preview_from_polys(h, preview,(0,0,0),1)
		
	exportOperations.export_polys_as_hpgl(lines, (int(output_image.shape[1]),int(output_image.shape[0])), timestamp + "4_black", wp, hp,"70_90",270,dx,dy)
	preview = imageOperations.create_preview_from_polys(lines, preview,(0,0,0),1)
	cv2.putText(preview,"Leertaste fuer neue Aufnahme", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0)
	cv2.imshow("tracant", preview)
 
 
	# update id and write to file
	cf["currentId"] += 1
	with open("counter.json", "w") as f:
		json.dump(cf, f)
 

def update_navigation(key,current_state,image_container,line_container):
	global bg_image
	if current_state == "image_select" or current_state == "style_select":
		temp = image_container[0]
		for i in range(len(image_container)):
			image_container[i] = temp.copy()
		if current_state == "image_select":
			current_state = "style_select"
			line_container = []
			#comfy
			cv2.imwrite("temp.png", image_container[key])
			imgs = comfyAdapter.run_comfy_workflow("temp.png")
   
			bg_image = cv2.imread("output/image_217.png", cv2.IMREAD_UNCHANGED)  
			bg_image = rgba_to_grayscale_white_background(bg_image)
			bg_image,mask = create_group_mask(bg_image)

			image_container[0] = cv2.imread("output/image_212.png", 0)
			image_container[1] = cv2.imread("output/image_214.png", 0)
			image_container[2] = cv2.imread("output/image_196.png", 0)
			image_container[3] = cv2.imread("output/image_196.png", 0)
			
			for img in image_container:
				t_mask = mask.copy()
				t_mask = cv2.resize(t_mask, (img.shape[1],img.shape[0]))
				img = np.maximum(img, t_mask)
   
			create_line_variations(image_container,line_container)


		elif current_state == "style_select":
			bg_image = cv2.resize(bg_image,(image_container[0].shape[1],image_container[0].shape[0]))
			finalize_image_group(bg_image,line_container[key])

			current_state = "export"
			
			#create_line_variations(image_container,line_container)
	return current_state,line_container

def create_group_mask(output_image):
	# get current position in grid from counter.json
	f = open("counter.json")
	cf = json.load(f)
	currentId = cf["currentId"]
	
	mask = output_image.copy()
	mask_img = layout["mask_img"]
	preview_img = layout["preview_img"]

	# get size and position settings for current element
	elemSettings = layout["grid"][currentId]

	# rotate images
	output_image = cv2.rotate(output_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
	mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

	x_px = elemSettings["xPx"]
	y_px = elemSettings["yPx"]
	w_px = elemSettings["widthPx"]
	h_px = elemSettings["heightPx"]

	

	# extract t_mask from mask_img to mask image
	t_mask = mask_img[y_px: y_px + h_px, x_px: x_px + w_px]
	t_mask = cv2.resize(t_mask, (output_image.shape[1], output_image.shape[0]))
	t_mask = np.invert(t_mask)
	
	# convert mask image to grayscale and mask it
	mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
 
 	# scale and translate mask for mask_img 
	mask_scaled_size = (w_px, h_px)
	mask_scaled = cv2.resize(mask, mask_scaled_size)
 
	# mask rgb image
	t_mask_rgb = cv2.cvtColor(t_mask, cv2.COLOR_GRAY2RGB)
	output_image = np.maximum(output_image, t_mask_rgb)

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
		mask_scaled, mask_img[y_px: y_px + h_px, x_px: x_px + w_px]
	)

	# export image and mask
	cv2.imwrite(config["outputFolder"] + "preview.bmp", preview_img)
	cv2.imwrite(config["outputFolder"] + "mask.bmp", mask_img)
	cv2.imwrite(config["outputFolder"] + str(currentId) + ".bmp", output_image)
 
	# rotate back
	output_image = cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE)
	t_mask = cv2.rotate(t_mask, cv2.ROTATE_90_CLOCKWISE)
	
	return output_image,t_mask

def rgba_to_grayscale_white_background(img):
	# Read the image with transparency
   # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	
	if img.shape[2] != 4:
		raise ValueError("Input image must have an alpha channel (RGBA format)")
	
	# Split the color and alpha channels
	bgr = img[:, :, :3]
	alpha = img[:, :, 3]
	
	# Create a white background (RGB)
	white_bg = np.ones_like(bgr) * 255
    
    # Reshape alpha to match RGB dimensions
	alpha_normalized = alpha / 255.0
	alpha_3d = np.stack([alpha_normalized] * 3, axis=-1)
    
    # Blend RGB image with white background based on alpha
	result = (bgr * alpha_3d + white_bg * (1 - alpha_3d)).astype(np.uint8)
    
	
	return result




##############
# main script
##############
# init mediapipe detectors for face and background subtraction
# init mediapipe detectors for face and background subtraction
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(
	model_selection=1, min_detection_confidence=0.5
)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# create grid
layout = gridCreation.create_perspective_grid(config)

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

		#image= cv2.imread("test.png")
		image = cv2.rotate(image, img_rotater[1])

		# the BGR image to RGB.
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


		# show output image
		if current_state == "cam" :
			cv2.imshow("tracant", image)
		elif current_state == "shooting" :

			# detect faces
			faces = face_detection.process(image)

			# continue if face detected
			if faces.detections:
				out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				blur_image = 9
				out = image.copy()
				out_blur = cv2.medianBlur(out, blur_image)
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
				image_container.append(out)

				if len(image_container)>=4:
					current_state = "image_select"
					preview = np.concatenate((image_container), axis=1)
					cv2.putText(preview,"Bild auswaehlen(n abbruch)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0)
					for i in range(len(image_container)):
						cv2.putText(preview,str(i), (20 + i*image_container[0].shape[1],100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
					cv2.imshow("tracant", preview)

			else: 
				image_container.append(image)
				if len(image_container)>=4:
					current_state = "image_select"
					preview = np.concatenate((image_container), axis=1)
					cv2.putText(preview,"Bild auswaehlen(n abbruch)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0)
					for i in range(len(image_container)):
						cv2.putText(preview,str(i), (20 + i*image_container[0].shape[1],100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
					cv2.imshow("tracant", preview)
				#current_state == "cam" 
		

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
	elif k == ord("p"):
		
		bg_image = cv2.imread("output/image_217.png", cv2.IMREAD_UNCHANGED)  
		bg_image = rgba_to_grayscale_white_background(bg_image)
		
		bg_image,mask = create_group_mask(bg_image)
  
		line_container = []
		image_container = []
		image_container.append(cv2.imread("output/image_212.png", 0))
		image_container.append(cv2.imread("output/image_214.png", 0))
		image_container.append(cv2.imread("output/image_196.png", 0))
		image_container.append(cv2.imread("output/image_196.png", 0))

		for img in image_container:
			t_mask = mask.copy()
			t_mask = cv2.resize(t_mask, (img.shape[1],img.shape[0]))
			img = np.maximum(img, t_mask)
		
		create_line_variations(image_container,line_container)
  
		bg_image = cv2.resize(bg_image,(image_container[0].shape[1],image_container[0].shape[0]))

		finalize_image_group(bg_image,line_container[3])
		current_state = "export"
  

	elif k & 0xFF == 27:
		break

	if (show_fps):
		print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


cap.release()
cv2.destroyAllWindows()
