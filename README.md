# Perception of a traffic scene

## Contents
1. [Calibration](#calibration)
2. Lane detection:
   - [Using perspective transformations](#find-middle-lane-line-with-perspective-transformations)



### Calibration

Calibration is an important step in computer vision because it helps to correct for any distortions or irregularities in the camera or imaging system being used. This is important because these distortions can affect the accuracy of any measurements or calculations made from the images captured by the camera. Calibration helps to ensure that the images captured are accurate and can be used for tasks such as object recognition, tracking, and measurement.

[Read more](../master/calibration)

![Undistorted_image](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/Undistorted_image.png)

### Find middle lane line with perspective transformations

Very popular method to obtain lane lines, but it may not work under different lighting conditions or sharply curved lanes. It requires a lot of tweaking of the threshold borders to remove white lines and light spots.

[Read more](../master/lane_detection/top-view)

![middle_line_perspective](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/top-view/detected_lane/video.gif)
