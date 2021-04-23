#######################################################################
# Program: Handwriting Recognition CNN
# Date: 4/20/21
# Author: Paul Pieper
# Dependencies: OpenCV, Tensorflow, Numpy, Sensehat, Picamera
# Model: Provided via:  https://keras.io/examples/vision/mnist_convnet/
#
# Desc:  Using the control stick on the sensehat (in any direction),
#        will take a picture using the PiCam. The image will be saved
#        into the CWD and will be 
#######################################################################

from sense_hat import SenseHat # SenseHat 
from picamera import PiCamera # Camera 
import time
import os, cv2, itertools # cv2 -- OpenCV
#from Keras.models import load_model # Keras Model Loading
#from keras.preprocessing.image import img_to_array
import tensorflow as tf # Tensorflow
import numpy as np # Numpy

# Assigning SenseHat and PiCamera
sense = SenseHat()  # Initialize SenseHat
camera = PiCamera() # Initialize Camera
camera.color_effects = (128,128) # Set camera to black and white mode

#Scaled&Thresh-Pilled 
ROWS = 28
COLS = 28
thresh = 160 #160 for Wht on Blk, 90 for Blk on Wht

# Read Image Function Using OpenCV
def read_negative_image(file_path):
	print('Applying filter to image...')
	#Reads image in using OpenCV grayscale
	sample_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	#Inverts image values
	sample_img = 255 - sample_img
	#Binary Filter
	sample_img = cv2.threshold(sample_img, thresh, 255, cv2.THRESH_BINARY)[1]
	#Write and Show Filtered Image
	print('Filter Applied!')
	cv2.imwrite('samp_BW.png', sample_img)
	cv2.imshow('Binary Input', sample_img)
	return cv2.resize(sample_img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)

# load and prepare the image
def load_image(filename):
	# load the image
	file = "sample_img.png"
	sample_img = read_negative_image(file)
	# convert to array
	sample_img = tf.keras.preprocessing.image.img_to_array(sample_img)
	# reshape into a single sample with 1 channel
	sample_img = sample_img.reshape(1, ROWS, COLS, 1)
	# prepare pixel data
	sample_img = sample_img.astype('float32')
	sample_img = sample_img / 255.0
	return sample_img
 
# load an image and predict the class
def run_network():
	# load the image
	sample_img = load_image('sample_img.png')
	# load model
	print('Running Neural Network...')
	model = tf.keras.models.load_model('keras_convnet_adam')
	# predict the class
	digit = model.predict_classes(sample_img)
	predictions = model.predict(sample_img)
	number = digit[0]
	print(number)
	print(predictions[0,number])
	output =  str(number)
	sense.show_letter(output)
	return

# Take a picture after Joystick Input
def say_cheese():
        # Capture Image, display
	print('Taking picture!')
	camera.capture('sample_img.png')
	camera.stop_preview()
	print('Picture Taken!')
	#cv2.imshow('Number?', sample_img) #, cmap="gray", vmin=0, vmax=255)
	#time.sleep(2)
	run_network()

# Main Function - Wait for Joystick to Run Code
##print('Waiting for joystick input from user...')
##sense.stick.direction_any = say_cheese

while True:
	print('Taking a picture...')
	say_cheese()
	pass


#while True:
#	pass  # This keeps the program running to receive joystick events
