from curses.textpad import rectangle
import numpy as np
import cv2 as cv


haar_cascade=cv.CascadeClassifier('har_frontface.xml')


people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

#features=np.load('features.npy',allow_pickle=True)
#labels=np.load('labels.npy')

face_recogniser=cv.face.LBPHFaceRecognizer_create()

face_recogniser.read('face_trained.yml')

img=cv.imread(r'/home/wajiha/Documents/opencvtutorial/media/Faces/val/ben_afflek/1.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("person",gray)

#detect the face in the image
face_rect=haar_cascade.detectMultiScale(gray,1.1,4)


for (x,y,w,h)  in face_rect:
    faces_roi=gray[y:y+h,x:x+h]

    label,confidence=face_recogniser.predict(faces_roi)
    print(label)
    print(confidence)

    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow("detectd",img)
cv.waitKey(0)
