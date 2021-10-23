import cv2
import imutils
import numpy as np
from opencv import run_avg
from opencv import segment
import matplotlib.pyplot as plt
from ml import prediction

bg = None

aWeight = 0.5

camera = cv2.VideoCapture(0)

top, right, bottom, left = 10, 350, 225, 590


num_frames = 0

i = 0 

while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        
        frame = imutils.resize(frame, width=700)

        
        frame = cv2.flip(frame, 1)

        
        clone = frame.copy()

        
        (height, width) = frame.shape[:2]

        
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            
            hand = segment(gray)

            
            if hand is not None:
                
                
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                thresholded = cv2.bitwise_not(thresholded)
                thresholded = np.stack((thresholded,)*3, axis=-1)
                gesture = prediction(thresholded)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(thresholded, 
                f"{prediction(thresholded)}", 
                (50, 50), 
                font, 1, 
                (0, 0, 255),
                2, 
                cv2.LINE_4)
                cv2.imshow("Thresholded", thresholded)

                
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1
        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        
        keypress = cv2.waitKey(1) & 0xFF

        
        if keypress == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()

cv2.destroyAllWindows()
