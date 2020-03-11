###USAGE__________________________________________________________________________
# python YOLO_main.py -i data/v03_plate.mp4 -o output/v03_plate_00.mp4 -y yolo-tiny -b 0 -d 1 -r 416
# python YOLO_main.py -i data/v09_plate.mp4 -o output/v09_plate_00.mp4 -y yolo-tiny -b 0 -d 1 -r 416 
# python YOLO_main.py -i data/plate4.jpg -o output/plate4.jpg -y yolo-coco -b 0 -d 1 -r 416

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

from plate_extraction import *

###ENTRADAS__________________________________________________________________________
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
ap.add_argument("-m", "--max", type=int, default=5,
	help="maximum number of detections per frame")
ap.add_argument("-b", "--blob", type=int, default=0,
	help="display or not blob")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="print information to debug")
ap.add_argument("-r", "--resolution", type=int, default=416,
	help="rxr resolution for yolo. 320, 416, 608, 832")
args = vars(ap.parse_args())

###INICIALIZAÇÃO_______________________________________________________________________

#LABELS
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#COLORS
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8") #COLORS: list of colors

#WEIGHTS and CONFIG
configName = "yolov3_"+str(args["resolution"])+".cfg"
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], configName])

#YOLO
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
if bool(args["debug"]): print(configPath)
if bool(args["debug"]): print(weightsPath)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #net: network object
ln = net.getLayerNames() #ln: List of names of neurons like 'conv_0', 'bn_0' 
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	#ln:list of output layers like ['yolo_82', 'yolo_94', 'yolo_106'] 

#VIDEO STREAM
# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

#FRAMES and FPS
# try to determine the total number of frames in the video file and fps
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
	#vs.set(cv2.CAP_PROP_FPS, 2000)
	fps = vs.get(cv2.CAP_PROP_FPS)
	if fps==1000: fps = 1
	print("[INFO] {} FPS in video".format(fps))
except:
	# an error occurred while trying to determine the total
	# number of frames in the video file
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

###LOOP_______________________________________________________________________________________
# loop over frames from the video file stream
elap_avg = []
plates = []
while True:
	start = time.time()
	#NEXT FRAME If grabbed is False, end of stream.
	(grabbed, frame) = vs.read()
	frame_org = frame.copy()
	if not grabbed:
		break

	#DIMENSIONS If the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	###PROCESSAMENTO_______________________________________________________________________________

	#BLOB
	#Construct a blob from the input frame
	#Perform a forward pass of YOLO
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (args["resolution"], args["resolution"]),
		swapRB=True, crop=False) #blob.type = np.darray (nimages,ncolors,H,W)

	#FORWARD PASS YOLO
	net.setInput(blob[:3]) #Sets the new value for the layer output blob.
	layerOutputs = net.forward(ln) #Runs forward pass for the whole network
		#layerOutputs: list of lists of detections

	#OUTPUT TREATMENT
	boxes = []
	confidences = []
	classIDs = []
	#Loop over each of the layer outputs
	for output in layerOutputs:
		#Loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	if bool(args["debug"]): print("confidences:", confidences) #[0.7333734035491943, 0.6028957366943359]
	if bool(args["debug"]): print("classIDs:", classIDs) #[2, 2] 
	if bool(args["debug"]): print("boxes:", boxes) #[[385, 109, 180, 65], [434, 116, 80, 51]] 

	#NON MAXIMA SUPPRESSION
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	#BOUNDING BOXES
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten()[:args["max"]]:
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			#CAR PLATE DETECTION
			plate = ''
			if x>0 and y>0 and w>100 and h>100:
				try:
					frame_car = frame[y:y+h, x:x+w, :]
					frame_car, frame_plate = extract_plate(frame_car)
					frame[y:y+h, x:x+w, :] = frame_car
					plate = extract_plate_chars(frame_plate)
					plates.append(plate)
				except:
					None
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.2f} {}".format(LABELS[classIDs[i]], confidences[i], plate)
			
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

	###DISPLAY________________________________________________________________________________
	#Resize the frame and convert it to grayscale (while still
	#retaining 3 channels) - possivelmente maior rapidez de exibição
	frame = imutils.resize(frame, width=800)
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#frame = np.dstack([frame, frame, frame])

	#TIME CALCULATIONS
	end = time.time()
	elap = (end - start)
	elap_avg.append(elap)
	cv2.putText(frame, "Frame time: {:.4f} s".format(elap), (15, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 3)
	cv2.putText(frame, "Frame avrg: {:.4f} s".format(np.mean(elap_avg)), (15, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 3)
	cv2.putText(frame, "Plates Detected:", (15, 90),
				cv2.FONT_HERSHEY_SIMPLEX, 1, [255,0,0], 3)
	for p in range(len(plates)):
		cv2.putText(frame, plates[p], (15, 120+p*20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,255], 1)

	# show the frame and update the FPS counter
	if args["blob"]:
		blob_img = np.zeros((args["resolution"],args["resolution"],3))
		blob_img[:,:,0] = blob[0,0,:,:]
		blob_img[:,:,1] = blob[0,1,:,:]
		blob_img[:,:,2] = blob[0,2,:,:]
		cv2.putText(blob_img, "Blob", (15, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 3)
		frame = img_glue(frame, blob_img)
	
	#OUTPUT
	cv2.imshow("Frame", frame)
	cv2.waitKey(1)

	###SALVANDO OUTPUT______________________________________________________________________
	# check if the video writer is None
	if writer is None:
		frame = frame_org.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = np.dstack([frame, frame, frame])
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	# write the output frame to disk
	if bool(args["output"]): writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
