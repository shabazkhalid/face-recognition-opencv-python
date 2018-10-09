import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')

conn = sqlite3.connect('students.db')
c = conn.cursor()
c.execute(''' CREATE TABLE IF NOT EXISTS students (id INT PRIMARY KEY NOT NULL, name text NOT NULL) ''')

def insert_or_update(Id, name):
    c.execute('select * from students where id = ' + Id)
    row = c.fetchone()
    if row is None:
        c.execute("insert into students values (?,?)",  (Id, name))
        conn.commit()
    else:
        print('record already exists, updating...')
        c.execute('UPDATE students SET name = ? WHERE id = ?', (name, Id))
        conn.commit()

i=0
Id = input('enter your ID: ')
name = input('enter your name: ')
insert_or_update(Id, name)

conn.close()

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=5, minSize=(100, 100))
    for(x,y,w,h) in faces:
        i=i+1
        cv2.imwrite("dataSet/face-" + Id + '.' + str(i) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-50:y+h+50,x-50:x+w+50])
        cv2.waitKey(100)
    if i >= 50:
        cam.release()
        cv2.destroyAllWindows()
        break
