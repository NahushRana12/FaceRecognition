import cv2
import sqlite3
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);

def insertOrUpdate(Id, Name):
    conn = sqlite3.connect("faceData.db")
    cmd = "SELECT * FROM person where Id = "+str(Id)
    cursor = conn.execute(cmd)
    rowExist = 0
    for row in cursor:
        rowExist = 1
    if(rowExist == 1):
        cmd = "UPDATE person SET Name "+str(Name)+" WHERE Id = "+str(Id)
    else:
        cmd = "INSERT INTO person(ID, Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
id = raw_input('enter user id')
name = raw_input('enter user name')
insertOrUpdate(id, name)
sampleNum = 0;

while(True):
    ret, img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);
    for(x, y, w, h) in faces:
        sampleNum=sampleNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100);
    cv2.imshow("Face", img);
    cv2.waitKey(1);
    if(sampleNum>40):
        break
 

cam.release()
cv2.destroyAllWindows()
