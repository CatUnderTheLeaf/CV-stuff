import os
import numpy as np
import cv2
from scipy.signal import find_peaks

from keras.models import load_model

model_dir_path = os.path.dirname(os.path.realpath(__file__)) + '\model'
model_path = os.path.join(model_dir_path, 'FCNN_model2.h5')
model = load_model(model_path)

def drawMiddleLine(cv_image):
    """Draw line on top of the image
        get several points of it
    
    Args:
        cv_image (OpenCV image): image to be drawn on
    
    Returns:
        np.array: image with line drawn on it
        np.array(x,y): a couple of points of the middle line
    """    
    binary = predict_binary(cv_image)
    blanks = np.zeros_like(binary).astype(np.uint8)
    # make 3 channel image
    lane_drawn = np.dstack((blanks, blanks, binary))
    ret_line_img = cv2.addWeighted(cv_image,  0.6, lane_drawn,  1, 0)
    return ret_line_img, []

def predict_binary(cv_image):
    small_img = np.array(cv_image)
    small_img = cv2.resize(small_img, (205, 154))
    small_img = small_img[1:small_img.shape[0]-1, 2:small_img.shape[1]-3]
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # return to initial size
    prediction = prediction[:,:,0]
    
    result = np.full((cv_image.shape[0]//2, cv_image.shape[1]//2), 0, dtype=np.uint8)
    result[1:cv_image.shape[0]//2-1, 2:cv_image.shape[1]//2-3] = prediction
    result = cv2.resize(result, (cv_image.shape[1], cv_image.shape[0]))
     
    return result