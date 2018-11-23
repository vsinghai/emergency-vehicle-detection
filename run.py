import numpy as np
import time
import cv2
from imutils.object_detection import non_max_suppression
import imutils
import argparse
from imutils import contours
from skimage import measure
from threading import Thread
import sys
if sys.version_info >= (3, 0):
	from queue import Queue
else:
	from Queue import Queue

points_of_interest = []
#class to handle multi threading for reading video files
class FileVideoStream:

	def __init__(self, path, queueSize=128):
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
 			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return

			# ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
 
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

#Point of Interest Class for found points of interest
average_weight = 0
class POI:
	def __init__(self, x, y, w, h, weight):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.weight = weight
		self.checked = False
		self.checker = False
		self.is_center = False


def FindLights(image):

	  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	  blurred = cv2.GaussianBlur(gray, (11, 11), 0)

	  #Image is now grayscaled and blurred 

	  # threshold the image to reveal light regions in the
	  # blurred image
	  #This operation takes any pixel value p >= 200 and sets it to 255 (white). 
	  # Pixel values < 200 are set to 0 (black).
	  thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

	  # perform a series of erosions and dilations to remove
	  # any small blobs of noise from the thresholded image
	  thresh = cv2.erode(thresh, None, iterations=2)
	  thresh = cv2.dilate(thresh, None, iterations=4)

	  # perform a connected component analysis on the thresholded
	  # image, then initialize a mask to store only the "large"
	  # components

	  labels = measure.label(thresh, neighbors=8, background=0)

	  #Initialize a mask to store only the large blobs
	  mask = np.zeros(thresh.shape, dtype="uint8")

	  # loop over the unique components
	  for label in np.unique(labels):
	      # if this is the background label, ignore it
	      if label == 0:
	          continue
	  
	      # otherwise, construct the label mask and count the
	      # number of pixels 
	      labelMask = np.zeros(thresh.shape, dtype="uint8")
	      labelMask[labels == label] = 255
	      #If blob exceeds threshold, blob is large enough to add to mask
	      numPixels = cv2.countNonZero(labelMask)
	  
	      # if the number of pixels in the component is sufficiently
	      # large, then add it to our mask of "large blobs"
	      if numPixels > 300:
	          mask = cv2.add(mask, labelMask)


	  #At this point, only large blobs will remain, no small ones

	  # find the contours in the mask, then sort them from left to
	  # right
	  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	      cv2.CHAIN_APPROX_SIMPLE)
	  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	  cnts = contours.sort_contours(cnts)[0]
	  
	  # loop over the contours
	  for (i, c) in enumerate(cnts):
	      ((cX, cY), radius) = cv2.minEnclosingCircle(c)
	      points_of_interest.append(POI(int(cX - radius), int(cY - radius), int(cX + radius), int(cY + radius), 1))


#Does scale and rotation independent template matching
#rotation only in the x-y plane, max_rotation is in degrees
def ScaledTemplateMatching(target, template, max_rotation):
	#edge detection on both images:
	gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
	gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	blurred_target = cv2.GaussianBlur(gray_target, (3, 3), 0)
	blurred_template = cv2.GaussianBlur(gray_template, (3, 3), 0)
	target_edges = cv2.Canny(blurred_target, 30, 200)

	for scale in np.linspace(0.08, 1.0, 30)[::-1]:
		# resize the image according to the scale
		newX, newY = template.shape[1] * scale, template.shape[0] * scale
		scaled_template = cv2.resize(gray_template, (int(newX), int(newY)))

		for angle in np.arange(0, max_rotation, 5):
			rotated = imutils.rotate(scaled_template, angle)
			template_edges = cv2.Canny(rotated, 30, 200)
		
			if target.shape[0] > newX and target.shape[1] > newY:
	
				result = cv2.matchTemplate(target_edges, template_edges, cv2.TM_CCOEFF_NORMED)
				#threshold variable for matches
				threshold = 0.3
				loc = np.where(result >= threshold)

				for pt in zip(*loc[::-1]):
					points_of_interest.append(POI(int(pt[0]), int(pt[1]), int(pt[0] + newX), int(pt[1] + newY), 3))

def CalculateConfidences(points):
	average_weight = 0
	for point in points:
		average_weight == point.weight
	average_weight = average_weight / len(points)

	confidences = []
	for point in points:
		confidences.append(point.weight / average_weight)


# constant needed for text detection method
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# uses the EAST text detector to detect any text in scene
# adds all detected text as a POI
def TextDetection(target, min_confidence):
	net = cv2.dnn.readNet('./east_text_detection.pb')
	width = target.shape[1]
	height = target.shape[0]
	new_width = 320
	new_height = 320
	resized_target = cv2.resize(target, (new_width, new_height))
	blob = cv2.dnn.blobFromImage(resized_target, 1.0, (new_width, new_height), 
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates tha
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		
		for x in range(0, numCols):

			if scoresData[x] < min_confidence:
				continue
	
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
		
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		width_ratio = width / float(new_width)
		height_ratio = height / float(new_height)

		startX = int(startX * width_ratio)
		startY = int(startY * height_ratio)
		endX = int(endX * width_ratio)
		endY = int(endY * height_ratio)
	
		points_of_interest.append(POI(startX, startY, endX, endY, 2))

template = cv2.imread('filled-star-of-life.jpg')

# Analyzes a frame or image and updates poi's
# returns count and locations of emergency vehicles
def analyzeFrame(image, is_video):
	points = []
	boxes = []
	global points_of_interest 
	points_of_interest = []

	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

	#ScaledTemplateMatching(image, template, 0)

	TextDetection(image, 0.7)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	count = 0
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])

			if CLASSES[idx] != "bus" and CLASSES[idx] != "car":
				continue

			count += 1
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			boxes.append((startX, startY, endX, endY))
			
			points.append((startX + (endX - startX) / 2, startY + (endY - startY) / 2))
		
	return count, points

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--list", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

images = args["list"].split(",")

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

#main loop

for image in images:  
	ext = image.split(".")[-1]
	if ext == "bmp":
		count, points = analyzeFrame(cv2.imread(image), False)
		print(count)
		print(points)
	elif ext == "mp4":
		counts = []
		all_points = []
		stream = FileVideoStream(image)
		stream.start()
		time.sleep(1.0)
		#vidcap = cv2.VideoCapture(image)
		#success, frame = vidcap.read()
		while stream.more():
			print("loading")
			frame = stream.read()
			count, points = analyzeFrame(frame, True)
			counts.append(count)
			all_points.append(points)
			stream.update()
		print(counts)
		print(all_points)


