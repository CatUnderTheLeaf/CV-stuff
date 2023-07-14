#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression

tracker_types = ['MIL','KCF','CSRT']
tracker_type = tracker_types[1]


class Tracker:
    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    def initTracker(self, image, initPose):
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(image, initPose)

    def detect(self, image):
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 6),
                padding=(2, 2), scale=1.19)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # pick = rects
        # print(weights)
        
        return pick

    def track(self, image):
        # Update tracker
        ok, bbox = self.tracker.update(image)
        (x, y, w, h) = bbox
        # print('tracking')
        return np.array([[x, y, x + w, y + h]])
    
    def runDetectionTracking(self, img_path):
        detected = False
        tracking = False
    
        # loop over the image paths
        for count, imagePath in enumerate(os.listdir(img_path)):
            # load the image and resize it to (1) reduce detection time
            # and (2) improve detection accuracy
            
            image = cv2.imread(os.path.join(img_path,imagePath))
            width=min(400, image.shape[1])
            height = int(image.shape[0] * width / image.shape[1])
            image = cv2.resize(image, (width, height))

            if detected:
                if not tracking:
                    print('init tracking')
                    # print(pick)
                    (xA, yA, xB, yB) = pick[0]
                    initPose = (xA, yA, xB-xA, yB-yA)
                    self.initTracker(image, initPose)
                    tracking = True
                pick = self.track(image)
                print(pick)
            else:
                pick = self.detect(image)
                print(pick)
                
            # draw the final bounding boxes        
            for (xA, yA, xB, yB) in pick:
                detected = True
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            if count%20==0:
                detected = False
                tracking = False   
                
            
            
            # show the output images        
            cv2.imshow("After NMS", image)
            cv2.waitKey(50)

        #     images.append(image)
        # imageio.mimsave(gif_name, images)

def main():

    full_path = os.path.dirname(os.path.realpath(__file__))
    
    import imageio
    images = []
    gif_name = os.path.join(os.path.dirname(full_path), 'one.gif')
    
    img_path = os.path.join(os.path.dirname(full_path),'pedestrian_detection','images','one')
    img_path = os.path.join(os.path.dirname(full_path),'pedestrian_detection','images','cars')
    img_path = os.path.join(os.path.dirname(full_path),'pedestrian_detection','images','bus')
    img_path = os.path.join(os.path.dirname(full_path),'pedestrian_detection','images','tunnel')
    # img_path = os.path.join(os.path.dirname(full_path),'pedestrian_detection','images','test')
    
    tracker = Tracker()
    tracker.runDetectionTracking(img_path)

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    
