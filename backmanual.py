import numpy as np
import cv2
from Tkinter import *
from PIL import Image
from PIL import ImageTk
from scipy import stats
import copy


# Just a silly counter to keep track of frames being
# generated, it can be used to save frames by giving them
# unique names in serial order. The counter will be increased
# by 1 as new frame will be read from video feed.
frameNumber = 0

# A boolean value tell if the frame is to be saved or not
saveFrame = False
# backdetect = cv2.BackgroundSubtractorMOG(history=1000,nmixtures = 5,backgroundRatio=0.7,noiseSigma=0);
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
backdetect = cv2.BackgroundSubtractorMOG2();

winName1 = "Live feed"
winName2 = "Background subtraction"

cv2.startWindowThread()
cv2.namedWindow(winName1)
# cv2.namedWindow(winName2)


cap1 = cv2.VideoCapture(0);

###########################################
def backgroundDetect():
    buf = 200;
    stride = 15;
    r = np.random.randint(0,stride,buf+1);
    print r
    t=0;

    start = 500;
    k = start+stride*buf;
    diff = np.zeros(1,np.float64);

    ret,frame = cap1.read();
    # cv2.imshow(winName1,frame)
    for i in range(k+1):
        ret,f = cap1.read(); # read the next video frame
        # cv2.imshow(winName1,f);
        if( i == 1):
            height,width = f.shape[:2];
            framesR = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            print framesR.shape
            framesG = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            framesB = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            f_1 = f;
            # diff(i) = 0;
        else:
            # diff(i) = sum(sum(abs(rgb2gray(f)-rgb2gray(f_1))>0.07))/(s(1)*s(2));
            f_1 = f;


        if (t<=buf and i>=start and r[t] == i%stride):
            # cv2.imshow('222',f);
            print t
            framesR[:,:,t] = f[:,:,0];
            framesG[:,:,t] = f[:,:,1];
            framesB[:,:,t] = f[:,:,2];
            t = t + 1;


        # %foreground = step(foregroundDetector, frame);

    background = f;
    temp=stats.mode(framesR,2)[0];
    # print temp;
    print temp.shape;
    background[:,:,0] =temp[:,:,0]
    temp=stats.mode(framesG,2)[0];
    # print temp;
    print temp.shape;
    background[:,:,1] =temp[:,:,0]

    temp=stats.mode(framesB,2)[0];
    print temp.shape;
    background[:,:,2] =temp[:,:,0]
    # print temp;
    # background[:,:,0] =
    # background[:,:,1] = stats.mode(framesG)[0];
    # background[:,:,2] = stats.mode(framesB)[0];


    # cv2.imshow(winName2,background);
    return background

# ###########################

g=backgroundDetect();


ggray= cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
#

while True:

    ret,frame2= cap1.read()
    fgray= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    foooo = abs(ggray - fgray)
    cv2.imshow('fpppp',foooo)
    foreground1 = (foooo > 5 )
    foreground1= np.asarray(foreground1,dtype=float)
    print foreground1
    # foreground1.astype(float)
    cv2.imshow(winName1,g)

    cv2.imshow('live',frame2)
    cv2.imshow('foregrounf',foreground1)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap1.release()
cv2.destroyWindow(winName1)
# cv2.destroyWindow(winName2)
