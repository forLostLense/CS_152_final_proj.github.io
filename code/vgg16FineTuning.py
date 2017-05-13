# PURPOSE: # PURPOSE: Running this program should train the top layers of our network 
# (everything but the VGG16 layers) on the faces in our data set, 
# trying to classify faces into 5 gender classes (each 20 years wide). The program will 
# create files containing the final weights.
# 
# USAGE: Run saveAgeData.py to create two files containing numpy arrays of our training and
# validation data. Make the variables trainDir and validationDir equal to the directory
# variable used in saveAgeData.py. 
#  
# ATTRIBUTIONS: The top-level model design and code structure were obtained here:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#

import numpy
import scipy.ndimage
import os
import matplotlib.pyplot as plt
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras import applications

from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, Activation
from keras.models import Model, Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import plot_model


# Variables
weightsPath = 'transferLearningWeights.h5'
trainDir = 'train_set'
validationDir = 'validation_set'
numTrain = 2000
numVal = 800
epochs = 50
batchSize = 16


def bottleneck():

	# Get our data
	trainData = numpy.load('data_' + trainDir + '.npy') / 255.0
	validationData = numpy.load('data_' + validationDir + '.npy') / 255.0

	# Pre-made network network
	model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224,3))
	
	# Without changing the weights, get the output the pre-trained model gives on our data
	trainBottleneck = model.predict(trainData, batch_size=batchSize, verbose=1)
	valBottleneck = model.predict(validationData, batch_size=batchSize, verbose=1)

	# Now train the top-level model
	trainTop(trainData, validationData)


def trainTop(trainData, validationData):

	trainLabels = numpy.load('ages_' + trainDir + '.npy')
	validationLabels = numpy.load('ages_' + validationDir + '.npy')

	# Top-level model
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, activation='relu',input_shape=(7,7,512)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(5, activation='softmax'))

	# Uncomment to pring diagram of architecture
	# plot_model(model, to_file='modelFrozen.png')

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd,
				  loss='categorical_crossentropy', 
				  metrics=['accuracy'])

	model.fit(trainData, trainLabels,
			  epochs=epochs,
			  batch_size=batchSize,
			  validation_data=(validationData, validationLabels))
	
	# Save weights so we can use them later
	model.save_weights(weightsPath)



if __name__ == "__main__":

	#Train model
	bottleneck()

