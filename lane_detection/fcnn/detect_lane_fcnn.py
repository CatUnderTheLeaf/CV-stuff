import os
import cv2
import glob
import laneFinderFCNN

test_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'\images'
save_path = os.path.dirname(os.path.realpath(__file__)) + '\detected_lane'

##################################################

# Visualize a diagram of the network

# from keras.models import load_model
# import visualkeras

# model_dir_path = os.path.dirname(os.path.realpath(__file__)) + '\model'
# model_path = os.path.join(model_dir_path, 'FCNN_model2.h5')
# model = load_model(model_path)

# visualkeras.layered_view(model, draw_volume=True, legend=True, scale_xy=1, to_file=save_path+'\model.png').show()
###################################################

images = glob.glob('lane_detection/images/*.jpg')

for fname in images:
    camera_img = cv2.imread(fname)
    line_img, img_waypoints = laneFinderFCNN.drawMiddleLine(camera_img)
    cv2.imwrite(os.path.join(save_path, 'line_'+ os.path.basename(fname)), line_img)

######################################
# For testing purposes

# camera_img = cv2.imread(images[230])
# line_img, img_waypoints = laneFinderFCNN.drawMiddleLine(camera_img)
# cv2.imwrite(os.path.join(save_path, 'hist.jpg'), line_img)

#######################################

#### uncomment to make a gif from images

# import imageio
# images = []
# detimages = glob.glob('**/fcnn/detected_lane/line_*.jpg')
# gif_name = os.path.join(save_path, 'video2.gif')
# for filename in detimages:
#     images.append(imageio.imread(filename))
# imageio.mimsave(gif_name, images)