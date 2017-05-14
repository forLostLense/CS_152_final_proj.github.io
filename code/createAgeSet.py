import numpy
import scipy.ndimage
import os


directory= "/Users/yvenica/Desktop/face_alignment/224_faces_new"


def calcAge(currMonth, currYear, birthMonth, birthYear):
    """
    Takes in current month & year and birth month & year 
    Returns age; if the information is invalid, returns -1 instead 
    """
	try:
		age = int(currYear) - int(birthYear)
		if (int(currMonth) < int(birthMonth)):
			age -= 1
		return age
	except:
		return -1

def process_data(num, directory):
    """
    Takes in the number of pictures per bucket and file directory 
    Creates the balanced dataset with num pictures per bucket
    """
    count_num = [0,0,0,0,0]
    for filename in os.listdir(directory):
        if sum(count_num) >= num * 5:
            return
        else:
            filename2=filename.replace('_','-')
            filename2=filename2.replace('.','-')
            fileWords = filename2.split('-')
            age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
            # check if each bucket has proper number of pictures
            if age != -1 and age < 100 and count_num[int(age / 20)] < num:
                count_num[int(age / 20)] += 1
                os.rename(directory + '/' + filename, '/Users/yvenica/Desktop/cnn/224_faces/train_set2/'+filename)
                print (age,count_num, sum(count_num), int(age / 20))
    print ("no")
    return

def check_data(directory):
    """
    Takes in the file directory of dataset
    Loops through all pictures in the directory and sort them into five buckets
    Return the list L(length 5) which stores the ages per bucket
    """
    L = [[],[],[],[],[]]
    for filename in os.listdir(directory)[1:]:
        filename2=filename.replace('_','-')
        filename2=filename2.replace('.','-')
        fileWords = filename2.split('-')
        print (fileWords)
        age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
        print (age)
        L[int(age / 20)].append(age)
    return L
L =check_data("/Users/yvenica/Desktop/cnn/224_faces/train_set2")

def label4(num, directory):
    """
    Takes in the number of pictures per bucket and file directory of dataset 
    Loops through the directory and add num label4 pictures to our dataset 
    """
    count= 0
    for filename in os.listdir(directory):
        if count >= num:
            return 
        else:
            filename2=filename.replace('_','-')
            filename2=filename2.replace('.','-')
            fileWords = filename2.split('-')
            age = calcAge(fileWords[3], fileWords[4], fileWords[2], fileWords[1])
            if age != -1 and int(age / 20) == 4:
                count += 1
                os.rename(directory + '/' + filename, '/Users/yvenica/Desktop/cnn/224_faces/label4/'+filename)
                print (age,count, int(age / 20))
    print ("no")
    return
#process_data(200, directory)
print (len(L[4]))

