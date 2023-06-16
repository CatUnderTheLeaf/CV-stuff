import os
import cv2
import numpy as np
import glob
import laneFinder

# matrices for perspective transformation
transform_matrix = np.array(
                            [[-1.44070347e+00, -2.24787583e+00,  5.03003158e+02],
                            [ 5.54569940e-16, -4.76928443e+00,  7.40727232e+02],
                            [ 1.37242848e-18, -1.09231360e-02,  1.00000000e+00]])

inverse_matrix =np.array(
        	            [[4.83440511e-01, -4.72483945e-01,  1.06809629e+02],
                        [ 1.49479041e-16, -2.09675060e-01,  1.55312027e+02],
                        [ 9.62443421e-19, -2.29030920e-03,  1.00000000e+00]])

test_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'\images'
save_path = os.path.dirname(os.path.realpath(__file__)) + '\detected_lane'

images = glob.glob('lane_detection/images/*.jpg')

for fname in images:
    camera_img = cv2.imread(fname)
    line_img, img_waypoints = laneFinder.drawMiddleLineWarp(camera_img, transform_matrix, inverse_matrix)
    cv2.imwrite(os.path.join(save_path, 'middle_line_'+ os.path.basename(fname)), line_img)

#### uncomment to make a gif from images

# import imageio
# images = []
# detimages = glob.glob('**/top-view/detected_lane/*.jpg')
# gif_name = os.path.join(save_path, 'video.gif')
# for filename in detimages:
#     images.append(imageio.imread(filename))
# imageio.mimsave(gif_name, images)