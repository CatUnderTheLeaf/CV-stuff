# Generate a PointCloud from stereo-pair image

0. Get a stereo pair images, eg. [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/) or [KITTI Stereo](https://www.cvlibs.net/datasets/kitti/eval_stereo.php)
   <p align="center">
    <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/left.png" width="400" title="left">
    <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/right.png" width="400" title="right">
  </p>
  
1. Convert a stereo pair images to grayscale and a little bit blur them
   <p align="center">
    <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/left.png" width="400" title="left_blur">
    <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/right.png" width="400" title="right_blur">
   </p>
  
2. Create a `StereoSGBM` or `StereoBM` object and compute a disparity map. In the code there are trackbars, so it will be easy to try different values for important parameters

![disparity_map](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/disparity_map.png)

3. To project image points to 3d you can:
   - use math
     - look at the diagram
     - from it can be derived formulas
    $$disparity = x - x' = \frac{Bf}{Z}$$
     - After Z is determined, X and Y can be calculated using the usual projective camera equations
    $$X = \frac{(col - centerCol)Z}{f}$$
    $$Y = \frac{(row - centerRow)Z}{f}$$
   - use `cv2.reprojectImageTo3D()`, but beforehand you need to calculate disparity-to-depth mapping matrix with `cv2.stereoRectify()`, don't forget to convert distance between cameras to meters for translation vector `T`
4. Save points as *.ply file. It can be easily viewed with [open3d](http://www.open3d.org/docs/0.9.0/tutorial/Basic/pointcloud.html) or [Meshlab](https://www.meshlab.net/)

<p align="center">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/side.png" title="pointcloud">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/front.png" title="pointcloud-front">
</p>
