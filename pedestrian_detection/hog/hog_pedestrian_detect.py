#!/usr/bin/env python

# We have object detection using keypoints, 
# local invariant descriptors, 
# and bag-of-visual-words models. 
# We have Histogram of Oriented Gradients. 
# We have deformable parts models. 
# Exemplar models. 
# And we are now utilizing Deep Learning with pyramids to recognize objects at different scales!




# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression

def main():

    full_path = os.path.dirname(os.path.realpath(__file__))
    
    import imageio
    images = []
    gif_name = os.path.join(os.path.dirname(full_path), 'one.gif')
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    
    img_path = os.path.join(os.path.dirname(full_path),'images','one')
    img_path = os.path.join(os.path.dirname(full_path),'images','cars')
    img_path = os.path.join(os.path.dirname(full_path),'images','bus')
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
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 6),
            padding=(2, 2), scale=1.19)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        # pick = rects
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        print(imagePath)
        if imagePath=='03m_12s_635379u.pgm':
            gif_name = os.path.join(os.path.dirname(full_path), 'false_boxes2.jpg')
            cv2.imwrite(gif_name, image)
        print(weights)
        # show the output images        
        cv2.imshow("After NMS", image)
        cv2.waitKey(10)

    #     images.append(image)
    # imageio.mimsave(gif_name, images)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    
