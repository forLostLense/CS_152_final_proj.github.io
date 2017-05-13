#
# PURPOSE: Running this program should train the network on the faces in our data set, 
# trying to classify faces by gender. The program wll create files containing the final 
# weights and graphs of the training process.
#
# USAGE: First run saveGData to generate two data files, one containing a matrix of 
# images, another creating a matrix of genders.  Place these files in the same 
# directory as gender2.py.  
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

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model


# Variables
trainDirectory = "gender_train_set"
validationDirectory = "gender_validation_set"
epochs = 20
batchSize = 10
cutoff = .5 # This tells us what proportion of our data set to use



def trainModel():

	# Convolutional model

	model = Sequential()
	model.add(Convolution2D(96, (7,7), activation='relu', input_shape=(224, 224, 3))) # Pics are 224 x 224 x 3
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	model.add(BatchNormalization(axis=1)) #TODO: IDK which axis is correct
	#model.add(Dropout(0.3))

	model.add(Convolution2D(256, (5,5), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(BatchNormalization(axis=1))
	#model.add(Dropout(0.3))

	model.add(Convolution2D(384, (3,3), activation='relu'))
	model.add(MaxPooling2D((3,3), strides=(2,2)))
	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1
		, activation='sigmoid'))

	# Uncomment this line to generate a graph of the model
	#plot_model(model, to_file='model2.png')

	sgd = SGD(lr=0.001, 
			  decay=1e-6, 
			  momentum=0.95, 
			  nesterov=True)

	model.compile(optimizer='rmsprop',
				loss='binary_crossentropy', 
				metrics=['accuracy'])


	# Load our data
	# The "split point" data is used to choose only a fraction of our data to run (if desired)

	trainData = numpy.load(trainDirectory + '_data.npy') / 255.0
	splitPoint = int(trainData.shape[0] * cutoff)
	trainData = trainData[:splitPoint]
	trainAges = numpy.load(trainDirectory + '_genders.npy')[:splitPoint]

	validationData = numpy.load(validationDirectory + '_data.npy') / 255.0
	splitPoint = int(validationData.shape[0] * cutoff)
	validationData = validationData[:splitPoint]
	validationAges = numpy.load(validationDirectory + '_genders.npy')[:splitPoint]

	# Train on the data, and save weights so we can use them later
	history = model.fit(trainData, trainAges,
			  epochs=epochs,
			  batch_size=batchSize,
			  validation_data=(validationData, validationAges))
	model.save_weights("genderWeightsReal.py")

	# Print out graphs of the loss/accuracy over time
	makePrettyPics(history)




def makePrettyPics(history):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	savefig('acc.png')

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	savefig('loss.png')



if __name__ == "__main__":
	trainModel()

