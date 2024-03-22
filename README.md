# Perception of a traffic scene

## Contents

1. Object Classification
   - [Classification of autonomous driving objects using transfer learning](#classification-with-transfer-learning)
2. Object Detection
   - [Run inference with pretrained object-detection models](#running-object-detector-inference)
3. Lane Detection:
   - [Lane Detection with OpenCV using perspective transformations](#lane-detection-with-perspective-transformations)
   - [Lane Detection with Deep Learning](#lane-detection-with-deep-learning)
4. Pedestrian Detection:
   - [Pedestrian detection with "Haar"-based Detectors](#pedestrian-detection-with-haar-based-detectors)
   - [Pedestrian detection with HOG and Linear SVM model](#pedestrian-detection-with-hog-and-linear-svm-model)
5. Object Tracking
6. Semantic and Instance Segmentation
7. [Calibration](#calibration)
8. [Generate a PointCloud from stereo-pair image](#generate-a-pointcloud-from-stereo-pair-image)
9. [Install Keras, Tensorflow and GPU support on WSL2](#ws2-installation)

### Classification with transfer learning

Before approaching object detectors I decided to work on classificators. Unfortunately I have not found a good dataset consisting only of cars/tracks/pedestrians/cyclists/etc. for such task in TensorFlow Datasets . But there are datasets for object detection in autonomous driving sphere. So I transformed object detection dataset into classification dataset. I applied Data Augmentation, so my model is robust to changes in input data such as lighting, cropping, and orientation. For transfer learning I used `mobilenet-v2` feature vector from TensorFlow Hub with frozen weights and just added `Dense` layer with my 8 classes ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram' and 'Misc') [Read more](../main/image_classification/classification_kitti_ds.ipynb)

![Classification](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/image_classification/output.png)

### Running object detector inference

Here I just wanted to see how good are existing pretrained models on my custom data. I used `ssd-mobilenet-v2` and `efficientdet-D0` pretrained on COCO 2017 dataset. I can say, that `efficientdet` is much slower than `ssd-mobilenet-v2` but detects more objects. [Read more](../main/object_detection/object_detection_inference.ipynb)

![ssd-mobilenet-v2](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/object_detection/detected/output.png)
![efficientdet-D0](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/object_detection/detected/output2.png)

### Lane Detection with perspective transformations

Very popular method to obtain lane lines, but it may not work under different lighting conditions or sharply curved lanes. It requires a lot of tweaking of the threshold borders to remove white lines and light spots. 

To detect all lane lines I have used Canny Edge Detector, for the middle line I have used a combination of color and gradient tresholds.

On the gifs it is definitely seen, that due to changes in light, sharp curves, (dis)apperaing border lane lines, lane lines are correctly detected in 90% cases. [Read more](../main/lane_detection/top-view)

|![line_perspective](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/top-view/detected_lane/video.gif)|![middle_line_perspective](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/top-view/detected_lane/video2.gif)|
| :---:     | :---:   |
|Detecting all lane lines         |      Detecting only yellow middle line|

### Lane Detection with Deep Learning

As can be seen from the gifs above, detecting lines with only OpenCV methods does not always give good results. Why not train a neural network that will detect lines despite poor lighting, tricky turns, and sidelines? Inspired by [Michael Virgo](https://github.com/mvirgo/MLND-Capstone) I decided to gather my own dataset and train a FCNN on it.

My model has a very simple architecture (diagram is made with [`visualkeras`](https://github.com/paulgavrikov/visualkeras)):

![fcnn_model](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/fcnn/detected_lane/model.png)

After 10 Epochs my model finished with MSE for training of 0.0040 and validation of 0.0039, which I think is pretty good. And here it is seen that FCNN is capable of detecting sharp curves in changing lighting and ignoring sidelines:

![fcnn_video](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/lane_detection/fcnn/detected_lane/video2.gif)

### Pedestrian detection with "Haar"-based Detectors

These detectors have been successfully applied to pedestrian detection in still images. The detectors only support frontal and back views but not sideviews. So there is a number of false alarms. I tried all three detectors: upper body detector, lower body detector and full body detector. [Read more](../main/pedestrian_detection/cascades)

![one_upper](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/cascades/detected/one_upper.gif)
![one_lower](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/cascades/detected/one_lower.gif)
![one_full](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/cascades/detected/one_full.gif)

### Pedestrian detection with HOG and Linear SVM model

Very easy method to detect pedestrians is to use pre-trained HOG and linear SWM model in OpenCV. Also I used non-maximum suppression to the bounding boxes to reduce number of overlaping boxes. After experiments with detecting parameters I can say that different images and cameras may require different parameter values to achieve the best results. And even when camera is the same lighting conditions, object size/scale/orientation and background clutter can tremendously effect the quality of detection. [Read more](../main/pedestrian_detection/hog)

![one_pedestrian](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/hog/detected/one.gif)
![tunnel_pedestrian](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/pedestrian_detection/hog/detected/tunnel.gif)

### Calibration

Calibration is an important step in computer vision because it helps to correct for any distortions or irregularities in the camera or imaging system being used. This is important because these distortions can affect the accuracy of any measurements or calculations made from the images captured by the camera. Calibration helps to ensure that the images captured are accurate and can be used for tasks such as object recognition, tracking, and measurement. [Read more](../main/calibration)

![Undistorted_image](https://github.com/CatUnderTheLeaf/scene_perception/blob/main/calibration/Undistorted_image.png)

### Generate a PointCloud from stereo-pair image

When we take photos, all depth information is lost due to perspective projection. Capturing a 3D point cloud is critical for self-driving technology because it allows accurate measurement of various objects on the road. It is possible to extract depth information from a stereo image pair. By comparing the left and right images, we can calculate a disparity and then convert it into a depth map. [Read more](../main/depth_image)

<p align="center">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/left.png" width="400" title="left">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/right.png" width="400" title="right">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/disparity_map.png" width="400" title="disparity map">
  <img src="https://github.com/CatUnderTheLeaf/scene_perception/blob/main/depth_image/images/side.png" width="400" title="pointcloud">
</p>

### WS2 Installation

Configuration for GPU usage: Ubuntu 22.04(WSL2), Python 3.10, CUDA Toolkit 12.2, cuDNN 8.9.7.29, tensorflow 2.15

1. Install [WSL2](https://docs.microsoft.com/windows/wsl/install) and Ubuntu 22.04
2. Open Ubuntu 22.04 in WSL2 and update Python and pip
```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3 #version 3.10
sudo apt-get install pip
pip install pip==21.3.1 # this version doesn't hate installation
```
3. Install fresh [NVIDIA® GPU drivers](https://www.nvidia.com/drivers)
4. Install [CUDA Toolkit 12.2 for WSL2](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
5. Download [cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x for Ubuntu 22.04](https://developer.nvidia.com/rdp/cudnn-archive)
6. Install cudnn
```
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.9.7.29-1+cuda12.2
sudo apt-get install libcudnn8-dev=8.9.7.29-1+cuda12.2
```
7. Install, create and activate `venv` (more user-friendly option to open your project in VS Code and there make virtualenvs)
```
sudo apt install python3.10-venv

python3 -m venv tensorflow2.15

source tensorflow2.15/bin/activate
```
8. Install tensorrt with `python3 -m pip install --upgrade tensorrt` to install tensorrt 8.6.1 (without it there was an installation error)
9. Finally install tensorflow with `pip install -U tensorflow[and-cuda]==2.15.0`
10. Edit `.bashrc`
```
# Export path variables !!! VERY IMPORTANT !!!
# Need to adapt to your python version:
export CUDNN_PATH="$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64"
# ...
export PATH="$PATH":"/usr/local/cuda/bin"
```
11. Check installation, should be 2.15.0 and `GPU` device listed
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
12. Install Keras and check its version, should be >= 3.0.0
```
pip install --upgrade keras
python3 -c "import keras; print(keras.__version__)"
```
> There are still errors with this installation:
> 
> E "Unable to register cuDNN/cuFFT/cuBLAS factory: Attempting to register factory for plugin cuDNN/cuFFT/cuBLAS when one has already been registered"
> 
> W "TF-TRT Warning: Could not find TensorRT"
> 
> I "could not open file to read NUMA node" - WSL2 is not build with NUMA support, as I understand
> 
> Previously needed this with Tensorflow 2.13
> ```
> # make sim link to cuda library for GPU usage
> # because it needs sim link instead of direct access to library file
> cd /usr/lib/wsl/lib
> sudo rm libcuda.so libcuda.so.1
> sudo ln -s libcuda.so.1.1 libcuda.so.1
> sudo ln -s libcuda.so.1 libcuda.so
> sudo ldconfig
> cd
> ```
