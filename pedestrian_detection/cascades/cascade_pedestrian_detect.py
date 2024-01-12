#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression

def main():

    full_path = os.path.dirname(os.path.realpath(__file__))
    
    full_body_cascade = cv2.CascadeClassifier()
    #-- 1. Load the cascades
    if not full_body_cascade.load(os.path.join(full_path,'haarcascade_fullbody.xml')):
    # if not full_body_cascade.load(os.path.join(full_path,'haarcascade_lowerbody.xml')):
    # if not full_body_cascade.load(os.path.join(full_path,'haarcascade_upperbody.xml')):
        print('--(!)Error loading face cascade')
        exit(0)

    import imageio
    images = []
    gif_name = os.path.join(full_path, 'detected', 'one_upper.gif')
    
    
    img_path = os.path.join(os.path.dirname(full_path),'images','one')
    # img_path = os.path.join(os.path.dirname(full_path),'images','cars')
    # img_path = os.path.join(os.path.dirname(full_path),'images','bus')
    # img_path = os.path.join(os.path.dirname(full_path),'images','tunnel')
    # img_path = os.path.join(os.path.dirname(full_path),'images','test')
    # loop over the image paths
    for imagePath in os.listdir(img_path):
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        
        image = cv2.imread(os.path.join(img_path,imagePath))
        width=min(400, image.shape[1])
        height = int(image.shape[0] * width / image.shape[1])
        image = cv2.resize(image, (width, height))

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.equalizeHist(image_gray)
        #-- Detect pedestrians
        rects = full_body_cascade.detectMultiScale(image_gray)
        
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # pick = rects
        # draw the final bounding boxes
        # for (xA, yA, xB, yB) in pick:
        #     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        for (x,y,w,h) in pick:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(imagePath)
        if imagePath=='03m_12s_635379u.pgm':
            gif_name = os.path.join(os.path.dirname(full_path), 'false_boxes2.jpg')
            cv2.imwrite(gif_name, image)
        # print(weights)
        # show the output images        
        cv2.imshow("After NMS", image)
        cv2.waitKey(10)

    #     images.append(image)
    # imageio.mimsave(gif_name, images)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    
