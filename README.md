# Perception of a traffic scene

## Contents
1. [Calibration](#calibration)
2. Lane detection:
   - [Lane Detection with perspective transformations](#lane-detection-with-perspective-transformations)
   - [Lane Detection with Deep Learning](#lane-detection-with-deep-learning)

### Calibration

Calibration is an important step in computer vision because it helps to correct for any distortions or irregularities in the camera or imaging system being used. This is important because these distortions can affect the accuracy of any measurements or calculations made from the images captured by the camera. Calibration helps to ensure that the images captured are accurate and can be used for tasks such as object recognition, tracking, and measurement. [Read more](../master/calibration)

![Undistorted_image](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/Undistorted_image.png)

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
