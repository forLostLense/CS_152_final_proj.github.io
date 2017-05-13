#
# PURPOSE: Get pic/age info from all the files in a directory. 
# Save data to files containing numpy arrays.
#
# USAGE: Modify the "directory" variable to point to the directories 
# containing the training and validation data. The output files will appear 
# in the same folder as saveAgeData.py and have the same name as the directory variable,
# with 'data_' or 'ages_ ' on the front.
#

import numpy
import scipy.ndimage
import os

directory = "validation_set"
imgWidth = 224
imgHeight = 224


def getData():
	"""Get pic/age info from all the files in a directory. 
	Save data to files containing numpy arrays."""

	numPics = int(len([name for name in os.listdir(directory)]))

	#Create empty numpy arrays
	data = numpy.zeros(shape=(numPics, imgHeight, imgWidth, 3)) #TODO: Standard amt?
	ages = numpy.zeros(shape=(numPics, 5)) #TODO: IS this the right way to do it?

	#Loop through files
	i=0
	badPics=0
	for filename in os.listdir(directory):
		if i >= numPics:
			break

		#Filename format: 42691_1901-11-03_1974.jpg
		filename2=filename.replace('_','-')
		filename2=filename2.replace('.','-')
		fileWords = filename2.split('-')
		
		age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
		
		if age > 0 and age <= 100:
			picData = scipy.ndimage.imread(directory + '/' + filename, flatten=False, mode="RGB")
			# Add another row to our data array
			data[i] = picData
			# Age array gets a one-hot vector, with a 1 in the appropriate age bin
			newRow = [0 for x in range(5)]
			newRow[(age-1)//20] = 1 
			ages[i] = newRow
			i += 1


	# Delete empty rows (there may be fewer rows than files if some file had bad data)
	badRows = [numPics - 1 - x for x in range(badPics)]
	data = numpy.delete(data, badRows, axis=0)
	ages = numpy.delete(ages, badRows, axis=0)

	# Save files containing the data
	numpy.save('data_' + directory + '.npy', data)
	numpy.save('ages_' + directory + '.npy', ages)

		


def calcAge(currMonth, currYear, birthMonth, birthYear):
	""" Given a birth month/year and another month/year, c
		alculate the years betwen them.
	"""
	try:
		age = int(currYear) - int(birthYear)
		if (int(currMonth) < int(birthMonth)):
			age -= 1
		return age
	except:
		return -1	


getData()	
