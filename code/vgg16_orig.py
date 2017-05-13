#
# PURPOSE: # PURPOSE: Running this program should train the top layers of our network 
# (everything but the VGG16 layers) on the faces in our data set, 
# trying to classify faces into 5 gender classes (each 20 years wide). The program will 
# create files containing the final weights.
# 
# USAGE: Run the program Modify the directory variable to point to the folder of faces in the data set.
#  
# ATTRIBUTIONS: The code for the VGG16 model was obtained here:
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
#

import numpy
import scipy.io
import os
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.vgg16 import VGG16




from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np

directory = "224_faces_small"
width = 224
height = 224

def VGG_16(weights_path=None):

	K.set_image_dim_ordering('th')
	model = Sequential()


	# Convolutional Model
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))

	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))

	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))

	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax')) # 100 corresponds to 100 age classes.

	if weights_path:
		model.load_weights(weights_path)

	return model



def train():

	# Create model
	model = VGG_16('vgg16_weights.h5')
	#NOTE: The weights file was too big to upload to Github.
	# It can be found here: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

	#Compile it
	sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	#Train it
	data = getData() #Returns an array: (data, ages)
	model.fit(data[0], data[1], nb_epochs=10, batch_size=10)
	return (model, data[0], data[1])



def getData():

	# Count how many pics we have so we know how long our array should be
	numPics = len([name for name in os.listdir(directory)])

	#Create empty numpy arrays
	data = numpy.empty(shape=(numPics, 3, width, height))
	ages = numpy.empty(shape=(numPics, 100))

	#Loop through files
	i=0
	badPics=0
	for filename in os.listdir(directory):

		#Filename format: 42691_1901-11-03_1974.jpg
		filename2=filename.replace('_','-')
		filename2=filename2.replace('.','-')
		fileWords = filename2.split('-')
		

		try :
			age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
			picData = plt.imread(directory + '/' + filename)
			# Add another row to our data array
			data[i] = np.swapaxes(picData, 0, 2) # We want (3, 224, 224), not (224, 224, 3)
			ages[i,0] = age
			i += 1
		except:
			badPics += 1

	# Delete empty rows
	badRows = [numPics - 1 - x for x in range(badPics)]
	data = numpy.delete(data, badRows, axis=0)
	ages = numpy.delete(ages, badRows, axis=0)
	
	return (data, ages)
	


def testFiles():
	directory = PATH

	#Loop through files
	for filename in os.listdir(directory):

		#Filename format: 42691_1901-11-03_1974.jpg
		filename2=filename.replace('_','-')
		filename2=filename2.replace('.','-')
		fileWords = filename2.split('-')
		age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
		picData = plt.imread(directory + '/' + filename)





def calcAge(currMonth, currYear, birthMonth, birthYear):
	age = int(currYear) - int(birthYear)
	if (int(currMonth) < int(birthMonth)):
		age -= 1
	return age

def getPic():
	return 
		



if __name__ == "__main__":

	#Train model
	trainResult = train()
	model = trainResult[0]
	data = trainResult[1]
	ages = trainResult[2]
	
	# Test trained model
	result=model.evaluate(data, ages)


	for filename in os.listdir(PATH):

		try:
			im = plt.imread(PATH + '/' + filename)
			out = model.predict(im)
			print("PREDICTION")
			print np.argmax(out)
		except:
			pass

