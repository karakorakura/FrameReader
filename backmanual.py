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

import numpy as np
import cv2
from Tkinter import *
from PIL import Image
from PIL import ImageTk
from scipy import stats
import copy
import algorithm.process_image as process_image


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
    buf = 50;
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
    # word frame and image have been used for one-another
    f1 = copy.deepcopy(frame2)

    fgray= cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    foooo = abs(ggray - fgray)
    cv2.imshow('fpppp',foooo)
    foreground1 = (foooo > 15 )
    foreground1= np.asarray(foreground1,dtype=float)
    cv2.imshow('binorig',foreground1)
    foooo2 = cv2.bitwise_and(f1,f1,mask=foreground1)
    foooo2 = cv2.cvtColor(foooo2,cv2.COLOR_BGR2GRAY)


    #########
    gray = cv2.medianBlur(foooo2,11)
    cv2.imshow('aftermedianBlur',gray)
    thresh1 = process_image.threshold_otsu(gray)
    cv2.imshow('afterotsu',thresh1)
    dilation = process_image.region_filling(thresh1)
    cv2.imshow('afterRegionfillingDilation',dilation)
    foreground1 = dilation
    kernel = np.ones((5,5),np.float)
    erosion = cv2.erode(foreground1,kernel,iterations=5)
    cv2.imshow('erosion',erosion)
    open1 = cv2.morphologyEx(erosion,cv2.MORPH_OPEN,kernel)
    cv2.imshow('open1',open1)
    close = cv2.morphologyEx(open1,cv2.MORPH_CLOSE,kernel)
    cv2.imshow('close',close)
    foreground1 = close
    ############
    cv2.imshow('frame2',frame2)
    maskedimage = cv2.bitwise_and(frame2,frame2,mask=foreground1)
    cv2.imshow('maskedimage',maskedimage)

    imYCR_CB = cv2.cvtColor(maskedimage,cv2.COLOR_BGR2YCR_CB)
    min_YCrCb = np.array([0,133,77], np.uint8)
    max_YCrCb = np.array([255,173,127], np.uint8)

    cv2.imshow('imYCR_CB',imYCR_CB)
    skinRegion = cv2.inRange(imYCR_CB,min_YCrCb,max_YCrCb)
    cv2.imshow('skinRegion',skinRegion)

    ##############
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the contour on the source image
    # This will convert skin color into black color
    facearea = 7000
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        # area can be configured
        if area > facearea:
            cv2.drawContours(f1, contours, i, (0, 0, 0), -1) # -1 fills the countour, else you can give outline thinkness




    # create gray scale image
    # f1small=f1[90:323,90:323]
    cv2.imshow('f1beforefinal',f1)
    gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('graybeforefinal',gray)

    # gray = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    # convert black to white and rest to black
    ret, final = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('final',final)
    ##############

    # print foreground1
    # foreground1.astype(float)
    cv2.imshow(winName1,g)

    cv2.imshow('live',frame2)
    cv2.imshow('foregrounf',foreground1)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap1.release()
cv2.destroyWindow(winName1)
# cv2.destroyWindow(winName2)

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
