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

###########################################
def backgroundDetect():
    buf = 400;
    stride = 15;
    r = np.random.randint(0,stride-1,buf);
    t=0;

    start = 500;
    k = start+stride*buf;
    diff = np.zeros(1,np.float64);
    cap1 = cv2.VideoCapture(0);
    ret,frame = cap1.read();
    for i in range(k+1):
        ret,f = cap1.read(); # read the next video frame
        cv2.imshow('111',f);
        if( i == 1):
            height,width = f.shape[:2];
            framesR = np.zeros((height,width,buf),dtype=np.float64);
            framesG = np.zeros((height,width,buf),dtype=np.float64);
            framesB = np.zeros((height,width,buf),dtype=np.float64);
            f_1 = f;
            # diff(i) = 0;
        else:
            # diff(i) = sum(sum(abs(rgb2gray(f)-rgb2gray(f_1))>0.07))/(s(1)*s(2));
            f_1 = f;


        if (t<=buf and i>=start and r[t] == i%stride):
            cv2.imshow('222',f);
            framesR[:,:,t] = f[:,:,0];
            framesG[:,:,t] = f[:,:,1];
            framesB[:,:,t] = f[:,:,2];
            t = t + 1;


        # %foreground = step(foregroundDetector, frame);

    background = f;
    background[:,:,1] = stats.mode(framesR);
    background[:,:,2] = stats.mode(framesG);
    background[:,:,3] = stats.mode(framesB);


    cv2.imshow('back',background);


# ###########################

backgroundDetect();
