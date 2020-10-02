#sccripts to collect image via robot
import numpy as np 
import cv2 
from skimage import io
from skimage import img_as_float
import time

cap = cv2.VideoCapture(1)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


img_count = 1
pre=time.time()

while(True):
     
    ret, frame = cap.read()
    show = cv2.imread('tmp.jpg')
    cv2.imshow('image_win', frame)
    now=time.time()
    if now-pre>1:
        pre=now
        cv2.imwrite('collected_img/'+str(img_count)+".jpg",frame)
        img_count+=1


    key = cv2.waitKey(1)

cap.release()

