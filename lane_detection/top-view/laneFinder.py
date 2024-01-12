# import os
import numpy as np
import cv2
from scipy.signal import find_peaks

def drawLinesWarp(cv_image, transformMatrix, inverseMatrix):
    """Draw line on top of the image
        get several points of it
    
    Args:
        cv_image (OpenCV image): image to be drawn on
    
    Returns:
        np.array: image with line drawn on it
        np.array(x,y): a couple of points of the middle line
    """    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    # return gray, []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # return blur, []
    gray = cv2.Canny(blur, 30, 80, True)
    # return gray, []
    
    warp_img = warp(gray, transformMatrix, inverseMatrix)

    # return warp_img, []
    pixels_line_img, nonzeropoints = find_line_pixels(warp_img, False, True)
    # return pixels_line_img, []

    line_img = np.zeros_like(cv_image)
    if (len(nonzeropoints)>0):
        for (x,y) in nonzeropoints:
            fit = get_polynomial(y, x)
            line_img, lane_points = draw_polyline(line_img, fit)
        # return line_img, []
        unwarp_img = warp(line_img, transformMatrix, inverseMatrix, top_view=False)
        ret_line_img = cv2.addWeighted(cv_image,  0.6, unwarp_img,  1, 0)
        return ret_line_img, np.array(lane_points)
    else:
        return cv_image, []

def drawMiddleLineWarp(cv_image, transformMatrix, inverseMatrix):
    """Draw line on top of the image
            get several points of it
        
        Args:
            cv_image (OpenCV image): image to be drawn on
        
        Returns:
            np.array: image with line drawn on it
            np.array(x,y): a couple of points of the middle line
    """    
    warp_img = warp(cv_image, transformMatrix, inverseMatrix)
    # return warp_img, []
    binary = yellow_treshold_binary(warp_img)
    # return np.dstack((binary, binary, binary)) *255, []
    pixels_line_img, nonzeropoints = find_line_pixels(binary,True, True)
    # return pixels_line_img, []
    line_img = np.zeros_like(cv_image)
    if (len(nonzeropoints)>0):
        for (x,y) in nonzeropoints:
            fit = get_polynomial(y, x)
            line_img, lane_points = draw_polyline(line_img, fit)
        # return line_img, []
        unwarp_img = warp(line_img, transformMatrix, inverseMatrix, top_view=False)
        ret_line_img = cv2.addWeighted(cv_image,  0.8, unwarp_img,  0.9, 0)
        return ret_line_img, np.array(lane_points)
    else:
        return cv_image, []

def warp(image, transformMatrix, inverseMatrix, top_view=True):      
    """wrap image into top-view perspective

    Args:
        image (cv2_img): image to transform
        top_view (bool, optional): warp into top-view perspective or vice versa. Defaults to True.

    Returns:
        cv2_img: warped image
    """        
    h, w = image.shape[0], image.shape[1]
    matrix = transformMatrix
    if not top_view:
        matrix = inverseMatrix
    if (matrix is None):
        print("before warp call set_transform_matrix()")
        return image
    else:
        birds_image = cv2.warpPerspective(image, matrix, (w, h))            
    return birds_image

def yellow_treshold_binary(image):
    """yellow color tresholding of an image,
    upper part of image is cut, it doesn't have relevant information

    Args:
        image (OpenCV image): image to be tresholded
    Returns:
        np.array: binary image
    """        
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_thresh=(140, 255)
    # b channel for yellow and blue colors
    b_channel = lab[:,:,2]
    lab_binary = np.zeros_like(b_channel)
    lab_binary[(b_channel >= lab_thresh[0]) & (b_channel <= lab_thresh[1])] = 1
    
    
    s_thresh=(30, 50)
    sx_thresh=(10, 100) #20, 100
    # Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) if (np.max(abs_sobelx)) else np.uint8(255*abs_sobelx)
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold light channel
    # in simulation h and s channels are zeros
    # if (np.max(s_channel) == 0):
    #     s_channel = l_channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the two binary thresholds
    yellow_output = np.zeros_like(sxbinary)
    # this outlines 3 lines
    yellow_output[(sxbinary == 1) & (lab_binary == 1) & (s_binary != 1)] = 1
    
    return yellow_output

def get_polynomial(y, x, ym_per_pix=1, xm_per_pix=1):
    """polynomial for a single line

    Args:
        y(array): y-coordinates of a line
        x(array): x-coordinates of a line
        ym_per_pix (int, optional): meters per pixel in y dimension. Defaults to 1.
        xm_per_pix (int, optional): meters per pixel in x dimension. Defaults to 1.

    Returns:
        array: polynomial of the line
    """  
    return np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

def find_line_pixels(image, middle=False, draw=False):
    """find middle line pixels in image with sliding windows

    Args:
        image (CVimage): tresholded binary image
        draw (bool, optional): draw rectangles on the image or not. Defaults to False.

    Returns:
        points: lines nonzero pixels
    """    
    
    points = []
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 35
    # Set minimum number of pixels found to recenter window
    minpix = 2
    line_treshold = 100

    blur = cv2.GaussianBlur(image, (21, 21), 0)

    # Take a histogram of the bottom half of the image
    # histogram = np.sum(image[image.shape[0]//6*5:,:], axis=0)
    histogram = np.sum(blur[image.shape[0]//3*2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((image, image, image)) *255
    # find 1-3 peaks of the lines
    peaks, properties = find_peaks(histogram, height =4, width=1, distance=100)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(histogram)
    # ax.plot(peaks, histogram[peaks], "x")
    # # ax.show()
    # import os
    # save_path = os.path.dirname(os.path.realpath(__file__)) + '\detected_lane'
    # fig.savefig(os.path.join(save_path, 'hist1.jpg'))
    
    # print(peaks, histogram[peaks])
    
    if (len(properties['peak_heights'])):     
        if middle:
            point = np.argmax(properties['peak_heights'])
            peaks = [peaks[point]]  
        # Peaks are the starting points for the lines
        for peak in peaks:
            x_base = peak
            
            # Set height of windows - based on nwindows above and image shape
            # window_height = np.int64((image.shape[0]//2+35)//nwindows)
            window_height = np.int64(image.shape[0]//nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = image.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated later for each window in nwindows
            x_current = x_base
            
            # Create empty lists to receive left and right lane pixel indices
            lane_inds = []
            lane_found = False
            
            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image.shape[0] - (window+1)*window_height
                win_y_high = image.shape[0]- window*window_height
                
                # Find the four below boundaries of the window 
                win_xleft_low = x_current - margin 
                win_xleft_high = x_current + margin                     
                # draw window rectangles
                if draw:
                    cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                    (win_xleft_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window 
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                # Append these indices to the lists
                lane_inds.append(good_left_inds)
                # If you found > minpix pixels, recenter next window on their mean position 
                # or stop windows
                if len(good_left_inds) > minpix:
                    x_current=np.int64(np.mean(nonzerox[good_left_inds]))
                    lane_found = True
                elif (len(good_left_inds) < minpix and lane_found):
                    break
                
            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            try:
                lane_inds = np.concatenate(lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                pass

            # Extract left and right line pixel positions
            x = nonzerox[lane_inds]
            y = nonzeroy[lane_inds] 
            out_img[y, x] = [0, 0, 255]
            if len(x)>line_treshold:
                points.append((x,y))
    return  out_img, points

def draw_polyline(img, fit):
    """draw polyline on the image

    Args:
        img (CVimage): image to be drawn on
        fit (_type_): lane line
    Returns:
        CVimage: image with a polyline
    """        
    draw_img = np.copy(img)
    ploty = np.linspace(0, draw_img.shape[0]-1, 10)
    
    fitx = get_xy(ploty, fit)
        
    all_points = [(np.asarray([fitx, ploty]).T).astype(np.int32)]
    cv2.polylines(draw_img, all_points, False, (0, 0, 255), 5)
    lane_points = np.asarray([fitx, ploty]).T
    
    return draw_img, lane_points

def get_xy(ploty, fit):
    """get x coordinates of the lane

    Args:
        ploty (array): y-coordinates
        fit (array): lane line

    Returns:
        tuple: y coordinates and left and right x coordinates
    """        
    try:
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        fitx = 1*ploty**2 + 1*ploty
        
    return fitx