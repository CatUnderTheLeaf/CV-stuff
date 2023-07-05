# Perception of a traffic scene

## Contents
1. [Calibration](#calibration)
2. [Generate a PointCloud from stereo-pair image](#generate-a-pointcloud-from-stereo-pair-image)
3. Lane detection:
   - [Lane Detection with perspective transformations](#lane-detection-with-perspective-transformations)
   - [Lane Detection with Deep Learning](#lane-detection-with-deep-learning)
4. Pedestrian detection:
   - [Pedestrian detection with HOG and Linear SVM model](#pedestrian-detection-with-hog-and-linear-svm-model)

### Calibration

Calibration is an important step in computer vision because it helps to correct for any distortions or irregularities in the camera or imaging system being used. This is important because these distortions can affect the accuracy of any measurements or calculations made from the images captured by the camera. Calibration helps to ensure that the images captured are accurate and can be used for tasks such as object recognition, tracking, and measurement. [Read more](../master/calibration)

![Undistorted_image](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/Undistorted_image.png)

### Generate a PointCloud from stereo-pair image

When we take photos, all depth information is lost due to perspective projection. Capturing a 3D point cloud is critical for self-driving technology because it allows accurate measurement of various objects on the road. It is possible to extract depth information from a stereo image pair. By comparing the left and right images, we can calculate a disparity and then convert it into a depth map. [Read more](../master/depth_image)

<p align="center">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/left.png" width="400" title="left">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/right.png" width="400" title="right">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/disparity_map.png" width="400" title="disparity map">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/side.png" width="400" title="pointcloud">
</p>

### Lane Detection with perspective transformations

Very popular method to obtain lane lines, but it may not work under different lighting conditions or sharply curved lanes. It requires a lot of tweaking of the threshold borders to remove white lines and light spots. 

To detect all lane lines I have used Canny Edge Detector, for the middle line I have used a combination of color and gradient tresholds.

On the gifs it is definitely seen, that due to changes in light, sharp curves, (dis)apperaing border lane lines, lane lines are correctly detected in 90% cases. [Read more](../master/lane_detection/top-view)

|![line_perspective](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/top-view/detected_lane/video.gif)|![middle_line_perspective](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/top-view/detected_lane/video2.gif)|
| :---:     | :---:   |
|Detecting all lane lines         |      Detecting only yellow middle line|

### Lane Detection with Deep Learning

As can be seen from the gifs above, detecting lines with only OpenCV methods does not always give good results. Why not train a neural network that will detect lines despite poor lighting, tricky turns, and sidelines? Inspired by [Michael Virgo](https://github.com/mvirgo/MLND-Capstone) I decided to gather my own dataset and train a FCNN on it.

My model has a very simple architecture (diagram is made with [`visualkeras`](https://github.com/paulgavrikov/visualkeras)):

![fcnn_model](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/fcnn/detected_lane/model.png)

After 10 Epochs my model finished with MSE for training of 0.0040 and validation of 0.0039, which I think is pretty good. And here it is seen that FCNN is capable of detecting sharp curves in changing lighting and ignoring sidelines:

![fcnn_video](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/fcnn/detected_lane/video2.gif)

### Pedestrian detection with HOG and Linear SVM model

Very easy method to detect pedestrians is to use pre-trained HOG and linear SWM model in OpenCV. Also I used non-maximum suppression to the bounding boxes to reduce number of overlaping boxes. After experiments with detecting parameters I can say that different images and cameras may require different parameter values to achieve the best results. And even when camera is the same lighting conditions, object size/scale/orientation and background clutter can tremendously effect the quality of detection. [Read more](../master/pedestrian_detection/hog)

![one_pedestrian](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/hog/one.gif)
![tunnel_pedestrian](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/hog/tunnel.gif)
