import cv2,os
import numpy as np
from PIL import Image 

recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = 'Classifiers/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascadePath)

def get_images_and_ids(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     # images will contains face images
     images = []
     # ids will contains the label that is assigned to the image
     ids = []
     for image_path in image_paths:
         # Read the image and convert to grayscale
         image_pil = Image.open(image_path).convert('L')
         # Convert the image format into numpy array
         image = np.array(image_pil, 'uint8')
         # Get the label of the image
         Id = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         #Id=int(''.join(str(ord(c)) for c in Id))
         # Detect the face in the image
         faces = faceCascade.detectMultiScale(image)
         # If face is detected, append the face to images and the label to ids
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             ids.append(Id)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     # return the images list and ids list
     return images, ids


images, ids = get_images_and_ids('dataSet')
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(ids))
recognizer.save('training.yml')
cv2.destroyAllWindows()