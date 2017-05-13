#
# PURPOSE: Running this program should train the network on the faces in our data set, 
# trying to classify faces into 5 gender classes (each 20 years wide). The program will 
# create files containing the final weights and graphs of the training process.
#
# USAGE: First run saveAgeData.py to generate two data files, one containing a matrix of 
# images, another creating a matrix of ages.  Place these files in the same 
# directory as this file. 
#
# ATTRIBUTIONS: We implemented the convolutional model from the paper 
# "Age and Gender Classification using Convolutional Neural Networks" (Levi & Tassner),
# found here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.9654&rep=rep1&type=pdf.
# 
# This Stack Overflow comment showed us how to display the accuracy/loss history graphs.
#



import numpy
import scipy.ndimage
import os
import matplotlib.pyplot as plt
import pydot
import graphviz
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

# Variables
trainDirectory = "train_set"
validationDirectory = "validation_set"
epochs = 40
batchSize = 10
cutoff = .5 # This tells us what proportion of our data set to use




def trainModel():

	# Convolutional Model
	model = Sequential()
	model.add(Convolution2D(96, (7,7), activation='relu', input_shape=(224, 224, 3))) # Pics are 224 x 224 x 3
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	model.add(BatchNormalization(axis=1))
	model.add(Dropout(0.3))

	model.add(Convolution2D(256, (5,5), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(BatchNormalization(axis=1))
	model.add(Dropout(0.3))

	model.add(Convolution2D(384, (3,3), activation='relu'))
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(5, activation='softmax')) # Length 5 corresponds to 5 output classes

	# Uncomment this line to generate a graph of the model
	###plot_model(model, to_file='model2.png')

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
	model.compile(optimizer=sgd,
				  loss='categorical_crossentropy', 
				  metrics=['accuracy'])

	# Load our data
	# The "split point" data is used to choose only a fraction of our data to run (if desired)

	trainData = numpy.load('data_' + trainDirectory + '.npy') / 255.0
	splitPoint = int(trainData.shape[0] * cutoff)
	trainData = trainData[:splitPoint]
	trainAges = numpy.load('ages_' + trainDirectory + '.npy')[:splitPoint]

	validationData = numpy.load('data_' + validationDirectory + '.npy') / 255.0
	splitPoint = int(validationData.shape[0] * cutoff)
	validationData = validationData[:splitPoint]
	validationAges = numpy.load('ages_' + validationDirectory + '.npy')[:splitPoint] 

	# Train on the data, and save weights so we can use them later
	history = model.fit(trainData, trainAges,
			  epochs=epochs,
			  batch_size=batchSize,
			  validation_data=(validationData, validationAges))
	model.save_weights("nnneuralWeights.py")

	makePrettyPics(history)
	


def makePrettyPics(history):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('acc.png')

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('loss.png')
		


def calcAge(currMonth, currYear, birthMonth, birthYear):
	try:
		age = int(currYear) - int(birthYear)
		if (int(currMonth) < int(birthMonth)):
			age -= 1
		return age
	except:
		return -1		


if __name__ == "__main__":

	#Train model
	trainModel()

