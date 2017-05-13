#
# PURPOSE: Get pic/gender info from all the files in a directory. 
# Save data to files containing numpy arrays.
#
# USAGE: Modify the last two lines of code to point to the directories 
# containing the training and validation data. The output files will appear 
# in the same folder and have the same name as he directory, with '_data.npy' 
# or '_genders.npy' appended to the end.
#


import numpy
import scipy.ndimage
import os

imgWidth = 224
imgHeight = 224



def getGender(filename, dataD):
	""" Takes the filename and a dictionary of genders
		Returns the person's gender, or -1 if not found."""
	
	# The numbers in the file name are the dictionary keys
	path = "" 
	for char in filename:
		if char != '_':
			path += char
		else:
			if path in dataD.keys():
				return dataD[path]
			else:
				return -1


def getData(directory):
	"""Get pic/gender info from all the files in a directory. 
	Save data to files containing numpy arrays."""

	# Count our pics so we know how long our array should be.
	numPics = int(len([name for name in os.listdir(directory)]))

	#Create empty numpy arrays
	data = numpy.zeros(shape=(numPics, imgHeight, imgWidth, 3))
	genders = numpy.zeros(shape=(numPics, 1))

	# Dictionary of gender data
	gDict = numpy.load("gender_info/genderDict.npy").item()

	#Loop through files
	i=0
	badPics=0
	for filename in os.listdir(directory):
		if i >= numPics:
			break

		gender = int(getGender(filename, gDict))
		if gender in [0,1]:
			# Add the new pic to our arrays
			picData = scipy.ndimage.imread(directory + '/' + filename, flatten=False, mode="RGB")
			data[i] = picData
			genders[i] = gender
			i += 1

	# Delete empty rows (there may be fewer rows than files if some file had bad data)
	badRows = [numPics - 1 - x for x in range(badPics)]
	data = numpy.delete(data, badRows, axis=0)
	genders = numpy.delete(genders, badRows, axis=0)

	# Save files containing the data
	numpy.save(directory + '_data.npy', data)
	numpy.save(directory + '_genders.npy', genders)



if __name__ == "__main__":
	getData("gender_info/gender_train_set")
	getData("gender_info/gender_validation_set")	
