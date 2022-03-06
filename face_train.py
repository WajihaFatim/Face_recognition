import os
import cv2 as cv
import numpy as np

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

DIR='/home/wajiha/Documents/opencvtutorial/media/Faces/train'

haar_cascade=cv.CascadeClassifier('har_frontface.xml')

features=[]
labels=[]


def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            face_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for(x,y,w,h) in face_rect:
                faces_roi=gray[y:y+h,x:x+w] #region of interest
                features.append(faces_roi)
                labels.append(label)


create_train()
print("training drain")

#print(len(features))
#print(len(labels))

features=np.array(features,dtype='object')
labels=np.array(labels)

face_recogniser=cv.face.LBPHFaceRecognizer_create()

#Train the recogniser on the features list and the labels list
face_recogniser.train(features,labels)


face_recogniser.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)