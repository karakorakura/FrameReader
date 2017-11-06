from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys

import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk
#from scipy import stats
import copy

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

frameNumber = 0
frameNumberRead=0
SignLabel = "4"
# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                  pygame.color.THECOLORS["blue"],
                  pygame.color.THECOLORS["green"],
                  pygame.color.THECOLORS["orange"],
                  pygame.color.THECOLORS["purple"],
                  pygame.color.THECOLORS["yellow"],
                  pygame.color.THECOLORS["violet"]]

def getDenoisedImage(frame):
    frameBlurred = cv2.medianBlur(frame, 7)
    # frameBlurred = cv2.medianBlur(frame, 11)
    cv2.imshow('after medianBlur',frameBlurred)
    return frameBlurred

def getSkinMaskedImage1(mask,frameYCrCb):
        # minRangeYCrCb = np.array([0,133,77], np.uint8)
        # minRangeYCrCb = np.array([255,173,127], np.uint8)
        # modified test skin colour
        minRangeYCrCb = np.array([16,133,77], np.uint8)
        maxRangeYCrCb = np.array([240,173,127], np.uint8)
        frameMasked   = cv2.bitwise_and(frameYCrCb,frameYCrCb,mask= mask)
        cv2.imshow('masked YcbCr',frameMasked)
        skinRegion = cv2.inRange(frameMasked,minRangeYCrCb,maxRangeYCrCb)
        cv2.imshow('skinRegion',skinRegion)
        return skinRegion

def thresholdOtsu(frame):
	ret,thresh1 = cv2.threshold(frame,75,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	return thresh1

def regionFilling(frame):
	#Have to Tune structuring element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	dilation = cv2.dilate(frame,kernel,iterations = 1)
	img_bw = 255*(frame> 5).astype('uint8')
	#opencv.imshow("dilate",dilation)
	return dilation

def getSkinMaskedImage2(mask,frame):
    frameMasked   = cv2.bitwise_and(frame,frame,mask= mask)
    converted = cv2.cvtColor(frameMasked, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    # tuned settings
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")
    skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)

    skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.medianBlur(skinMask, 5)
    return skinMask

def getColorDefinedFrame(frame):
    frameYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    minRangeYCrCb = np.array([16,133,77], np.uint8)
    maxRangeYCrCb = np.array([240,173,127], np.uint8)
    # frameMasked   = cv2.bitwise_and(frameYCrCb,frameYCrCb,mask= mask)
    # cv2.imshow('masked YcbCr',frameMasked)
    mask = cv2.inRange(frameYCrCb,minRangeYCrCb,maxRangeYCrCb)
    mask = getDenoisedImage(mask)
    skinRegion = cv2.bitwise_and(frame,frame,mask= mask)
    cv2.imshow('skinRegion',skinRegion)
    return skinRegion








class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data
        self._bodies = None


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);

        # Right Arm
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_rectangle(self,frame):
        #rectangle
        winName  = "Rectangle"
        cv2.startWindowThread()
        cv2.namedWindow(winName)

        while True:
            cv2.imshow('live',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(winName)

    def run(self):
        global frameNumber
        global frameNumberRead
        global SignLabel
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            # --- Game logic should go here

            # --- Getting frames and drawing
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                frame1 = copy.deepcopy(frame)

                # frameNumber+=1
                print(frameNumber)
                print (frame.shape)
                print(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height)
                frame1=np.reshape(frame1,(1080,1920,4))
                print (frame1.shape)
                winName  = "Rectangle"
                cv2.startWindowThread()
                cv2.namedWindow(winName)
                frame2 = frame1[:,:,0:4]
                frame3 = copy.deepcopy(frame2)
                print(frame2.shape)
                # frameNumber
                # cv2.imwrite("/output1/frame-"+str(frameNumber)+".png", sFrame1)




                self.draw_color_frame(frame, self._frame_surface)


                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked:
                        continue

                    joints = body.joints
                    # convert joint coordinates to color space
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])

                    tiplx = int(joint_points[PyKinectV2.JointType_HandTipLeft].x)
                    tiply = int(joint_points[PyKinectV2.JointType_HandTipLeft].y)

                    tiprx = int(joint_points[PyKinectV2.JointType_HandTipRight].x)
                    tipry = int(joint_points[PyKinectV2.JointType_HandTipRight].y)

                    wristlx = int(joint_points[PyKinectV2.JointType_WristLeft].x)
                    wristly = int(joint_points[PyKinectV2.JointType_WristLeft].y)

                    wristrx = int(joint_points[PyKinectV2.JointType_WristRight].x)
                    wristry = int(joint_points[PyKinectV2.JointType_WristRight].y)

                    cv2.rectangle(frame2,(tiplx-100,tiply-100),(wristlx+100,wristly+100),(0,255,0),1)
                    cv2.rectangle(frame2,(tiprx-100,tipry-100),(wristrx+100,wristry+100),(0,255,0),1)
                    cv2.imshow(winName,frame2)



                    saveFrame = True
                    min_xr = min(tiprx,wristrx)
                    max_xr = max(tiprx,wristrx)
                    min_yr = min(tipry,wristry)
                    max_yr = max(tipry,wristry)

                    min_xl = min(tiplx,wristlx)
                    max_xl = max(tiplx,wristlx)
                    min_yl = min(tiply,wristly)
                    max_yl = max(tiply,wristly)

                    if max_xl + 100 > min_xr - 100 and ( (max_yl > min_yr and max_yl < max_yr) or (min_yl < max_yr and min_yl > min_yr) ):
                        minx_final = min(min_xl,min_xr) +10;
                        maxx_final = max(max_xl,max_xr) -10;
                        miny_final = min(min_yl,min_yr) +10;
                        maxy_final = max(max_yl,max_yr) -10;

                    else :
                        minx_final = min_xr
                        maxx_final = max_xr
                        miny_final = min_yr
                        maxy_final = max_yr


                    #print(min_x,max_x,min_y,max_y)
                    frameNumberRead = frameNumberRead + 1
                    if saveFrame and frameNumberRead%8==0:

                        if minx_final-100>0 and miny_final-100>0:

                            framesave = frame3[miny_final-100:maxy_final+100,minx_final-100:maxx_final+100,:]
                            print(tiprx-100,wristrx+100,tipry-100,wristry+100)
                            sFrame = framesave
                            #sFrame = cv2.resize(framesave, (200, 200))
                            print (sFrame.shape)
                            sFrame1 = framesave[:,:,0:3]
                            sFrame1s = getColorDefinedFrame(sFrame1)
                            print (sFrame1.shape)

                            # name1 = r"F:\sem 6\capstone\Divanshu\Sign-Language-to-Speech-master\Live-feed-analyzer\output\frame-"+str(frameNumber)+".tiff"
                            # name2= r"F:\sem 6\capstone\Divanshu\Sign-Language-to-Speech-master\Live-feed-analyzer\output1\frame-"+str(frameNumber)+".png"
                            # SignLabel = "b"

                            name4D = r".\output4D\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"4d.tiff")
                            name3D = r".\output3D\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"3d.png")
                            name3Ds = r".\output3Ds\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"3ds.png")
                            #
                            # name4D = r".\output4D\"   + SignLabel + r"." +  str(frameNumber) + "."+"4d.tiff"
                            # name3D = r".\output3D\"   + SignLabel + r"." + str(frameNumber) + "."+"3d.png"
                            # name3Ds = r".\output3Ds\" + SignLabel + r"." + str(frameNumber) + "."+"3ds.png"

                            print(name4D,name3D,name3Ds)
                         # resize to 100 x 100 to save
                        # sFrame = cv2.resize(final, (100, 100)) # resize to 100 x 100 to save
                            #cv2.imshow('hello',framesave)
                            #cv2.imshow('hello',sFrame1)
                            cv2.imwrite(name4D, sFrame)
                            cv2.imwrite(name3D, sFrame1)
                            cv2.imwrite(name3Ds, sFrame1s)
                            frameNumber = frameNumber + 1



            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();
