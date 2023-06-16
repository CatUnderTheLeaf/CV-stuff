### Camera calibration

1. Print a [checkboard](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration?action=AttachFile&do=view&target=check-108.pdf).

2. Make 10-15 camera images in different angles, distance from camera and position in the image.

3. Run a simple [calibration code](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/calibrate.py)
![checkboards](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/checkboards.png)

4. Camera matrix, projection matrix and distortion coefficients should be printed in terminal.
```
# camera_matrix
[[202.17872655   0.         206.74570367]
 [  0.         200.93651591 153.25419243]
 [  0.           0.           1.        ]]
 
 # distortion_coefficients
[[-3.27310350e-01  1.18817621e-01 -8.73818848e-05  1.03891840e-03
  -2.04676122e-02]]
  
  # projection_matrix
[[201.68559265   0.         206.24143806   0.        ]
 [  0.         200.28413391 152.75661769   0.        ]
 [  0.           0.           1.           0.        ]]
```

5. Now the image can be undistorted.

![Undistorted_image](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/Undistorted_image.png)
