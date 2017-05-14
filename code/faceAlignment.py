# reference: http://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

# Image processing libraries
import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import cv2
# detect the face
def detect_faces(image):
    """
    Takes in the image file 
    Returns the frame of detected face 
    """
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames


def faceAlign():
    """
    Align and crop face from wiki_dataset
    """
    # Enter the wiki folder
    import os
    output_F = '/Users/yvenica/Desktop/face_alignment/224_faces_new'
    wiki_F = '/Users/yvenica/Desktop/wiki2'
    folder_L = os.listdir(wiki_F)[1:]
    #loop through the folders
    for folder in folder_L:
        os.chdir(wiki_F+'/'+folder)  # up one directory
        print (folder)
        image_L = os.listdir()[1:] # a list of images
        #loop through the images
        for imageN in image_L:
            img_path = wiki_F+'/'+folder+'/'+imageN
            try:
                image = io.imread(img_path)
                os.remove(img_path)
                width, height = len(image[0]), len(image)
                detected_faces = detect_faces(image)
                # Crop faces and plot
                for n, face_rect in enumerate(detected_faces):
                    # 40% margin
                    c_width = face_rect[2] - face_rect[0]
                    c_height = face_rect[3] - face_rect[1]
                    width_margin = width * 0.2
                    height_margin = height * 0.2
                    n_face_rect = (max(0,face_rect[0] - width_margin), max(0,face_rect[1] - height_margin), min(width, face_rect[2]+width_margin), min(height, face_rect[3]+height_margin))
                    face = Image.fromarray(image).crop(n_face_rect)
                    face = face.resize((224,224))
                    opt_path = output_F + '/' + imageN
                    face.save(opt_path)

            except:
                pass

faceAlign()
