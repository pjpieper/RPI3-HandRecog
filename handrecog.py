#######################################################################
# Program: Handwriting Recognition CNN
# Date: 4/20/21
# Author: Paul Pieper
# Dependencies: OpenCV, Tensorflow, Numpy, Sensehat, Picamera
# Model: Provided via:  https://keras.io/examples/vision/mnist_convnet/
#
# Desc:  After startup, a live image preview will show in a new window.
#	 Following this, the system will then take image captures
#	 using the "C" key, and assessing each as to what digit it
#	 most resembles, while printing out a confidence value. You can
#	 also perform a "continuous capture" if you hold down "C". 
#	 Pressing "Q" will quit the application.
#######################################################################

from sense_hat import SenseHat # SenseHat 
from picamera import PiCamera # Camera 
from skimage import img_as_ubyte
from skimage.color import rgb2gray
import time
from time import sleep
import datetime
import argparse
import imutils
from imutils.video import VideoStream
import os, cv2, itertools # cv2 -- OpenCV
import tensorflow as tf # Tensorflow
import numpy as np # Numpy

# Assigning SenseHat and PiCamera
sense = SenseHat()  # Initialize SenseHat
#camera = PiCamera() # Initialize Camera
#camera.color_effects = (128,128) # Set camera to black and white mode

#Scaled&Thresh-Pilled 
ROWS = 28
COLS = 28

#Load model
print('Loading Neural Network...\n')
model = tf.keras.models.load_model('mnist_trained_model.h5')
#model = tf.keras.models.load_weights('keras_convent_adam')

# Construct arguments for PiCamera.
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1, help="Should the PiCam be used?")
args = vars(ap.parse_args())
 
# Initialize VideoStream from PiCamera
vidStream = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# Read Image Function Using OpenCV and Create Binary Output
def read_negative_image(sample_img):

	print('Applying filter to image...')

	#Reads image in using OpenCV grayscale the converts to uint8
	img_gray =  rgb2gray(sample_img)
	img_gray8 = img_as_ubyte(img_gray)

	#Binary Filter and Scaling using the OTSU method
	(thresh, img_bw) = cv2.threshold(img_gray8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	img_resized = cv2.resize(img_bw,(28,28))

	#Invert the colors
	img_bw_invert = 255 - img_resized
	img_final = img_bw_invert.reshape(1,ROWS,COLS,1)
	#Write and Show Filtered Image (For Testing Purposes)
	#print('Filter Applied!')
	#cv2.imwrite('samp_BW.png', sample_img)
	#cv2.imshow('Binary Input', sample_img)

	run_network(img_final)

# Predict the classification of the image.
def run_network(img_final):

	#Array printout of each possible digit.
	answer = model.predict(img_final)
	print(answer)

	#Digit with the greatest possibility is the prediction
	prediction = answer[0].tolist().index(max(answer[0].tolist()))
	print('The predicted digit is: ', prediction)

	#Get confidence value
	confidences = np.squeeze(model.predict_proba(img_final))
	topClass = np.argmax(confidences)
	topConf = confidences[topClass]
	print(topClass)
	print(confidences)

	#Update Sensehat LED display when over the 50% threshold
	if topConf > .5:

		print('Confidence is: ', topConf)
		ledDigit = str(prediction)
		sense.show_letter(ledDigit)

##########################################################
# Main Function - Takes pictures when 'C' is pressed.
#		  Quits function when 'Q' is pressed.
#                 Waits for keyboard input to break loop.
##########################################################

def main():
	#Updates frames from VideoStream to Window Preview
	while True:
		try:

			#Resizes image input from VideoStream to 800px wide, auto adjust height.
			frame = vidStream.read()
			frame = imutils.resize(frame, width=400)
		 
			#Creates a timestamp on the image.
			timestamp = datetime.datetime.now()
			ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
			cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.35, (0, 0, 255), 1)
		 
			#Show frame in Window Preview
			cv2.imshow("Window Preview", frame)
			key = cv2.waitKey(1) & 0xFF
		 
			#Key catch statement. Refer to Main Function comment for controls.
			if key == ord("q"):
				break
				#KillAllWindows.exe
				cv2.destroyAllWindows()
				vidStream.stop()
			elif key == ord("c"):
				cv2.imwrite("number.jpg", frame)  
				sample_img = cv2.imread("number.jpg")
				read_negative_image(sample_img)
			else:
				pass
				
		except KeyboardInterrupt:
			#KillAllWindows.exe
			cv2.destroyAllWindows()
			vidStream.stop()
			

if __name__=="__main__":
	main()

