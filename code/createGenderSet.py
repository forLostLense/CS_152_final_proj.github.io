import os
import numpy as np
def getGenderData():
	"""
	Loops through gender text file (id : gender) and full path text file (fullpath : id).
	Returns a dictionary whose key is full path, value is gender (0 for female, 1 for male)
	"""

	# Read text files
	with open('wiki_gender.txt') as f:
	    content = f.readlines()
	
	gender_labelL= [x.strip() for x in content] 

	with open('full_path.txt') as f:
	    content = f.readlines()
	pathS= ''.join(content)


	# analyze the path
	read_bol = False
	path = ""
	full_pathL = []
	for char in pathS:
		if char == '/':
			read_bol = True
		if read_bol == True:
			if char == '_':
				read_bol = False
				full_pathL.append(path)
				path = ""
			else:
				if char != '/':
					path += char

	# match the dictionary key with corresponding value
	dataD = {}
	for idx in range(len(full_pathL)):
		dataD[full_pathL[idx]] = gender_labelL[idx]
	return dataD

def getGender(filename, dataD):
	"""
	Takes in the filename and data dictionary 
	Returns the corresponding gender value, if the path is not valid,
	returns -1 
	"""
	path = ""
	for char in filename:
		if char != '_':
			path += char
		else:
			if path in dataD.keys():
				return dataD[path]
			else:
				return -1

def createDataset(num, directory, dataD):
	"""
	Takes in the number of pictures per gender, the file directory of saving 
	pictures and data dictionary 
	Writes out a npy format file which stores the gender data dictionary
	"""
    count_num = [0,0]
    for filename in os.listdir(directory):
        if sum(count_num) >= num * 2:
            return
        else:
            gender = getGender(filename, dataD)
            if gender != -1 and gender != 'NaN' and count_num[int(gender)] < num:
                count_num[int(gender)] += 1
                os.rename(directory + '/' + filename, '/Users/yvenica/Desktop/cnn/224_faces/gender_validation_set/'+filename)
                print (gender,count_num, sum(count_num))
    return

dataD = getGenderData()
# directory= "/Users/yvenica/Desktop/face_alignment/224_faces_new"
# createDataset(100, directory, dataD)
np.save('genderDict.npy', dataD)
