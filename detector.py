import cv2
import os
import numpy as np
from PIL import Image
import pickle
import sqlite3

conn = sqlite3.connect('students.db')
c = conn.cursor()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training.yml')
cascadePath =  'Classifiers/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture('video.ogv')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, minNeighbors=5)


    for(x,y,w,h) in faces:
        prediction, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (255,255,0), 2)

        c.execute('select name from students where id = ' + str(prediction))
        prediction = c.fetchone()

        if confidence <= 65:
            cv2.putText(im, prediction[0], (x,y+h), font,1,(255,255,255),2)
        else:
            cv2.putText(im, 'unknown', (x,y+h), font,1,(0,0,255),2)
        
        print(prediction[0], confidence)

    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    cv2.imshow('im',im)
    cv2.waitKey(10)
conn.close()