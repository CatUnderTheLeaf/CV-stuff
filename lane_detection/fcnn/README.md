# Find middle lane line with Deep Learning

1. I have collected images using `rosbag` and made binary-mask label images using [Lane Detection with perspective transformations](/lane_detection/top-view),
   sorted out images where lines were not correctly detected due to poor light conditions etc, and got 6095 images and corresponding labels, 150 x 200 x 3 size.
   This dataset was divided into training and validation sets in proportion 90%:10%

2. Made a `U-Net`-like network with a following architecture:
   > kernel = (3, 3)
   > 
   > padding = 'valid'
   > 
   > strides = (1,1)
   >
   > pool_size = (2, 2)
   > 
   > activation = 'relu'
   > 
![architecture](/lane_detection/fcnn/detected_lane/model.png)

| Layer | Output Shape | Additional | 
|  ---  | -------- | ------ |
| BatchNormalization | (None, 152, 200, 3) |   |
| Conv2D  |  (None, 150, 198, 8) |   |
| Conv2D | (None, 148, 196, 16) |   |
| MaxPooling2D | (None, 74, 98, 16) |   |
| Conv2D | (None, 72, 96, 16) | dropout |
| Conv2D | (None, 70, 94, 32) | dropout |
| Conv2D | (None, 68, 92, 32) | dropout |
| MaxPooling2D | (None, 34, 46, 32)  |   |
| Conv2D | (None, 32, 44, 64) | dropout |
| Conv2D | (None, 30, 42, 64) | dropout |
| MaxPooling2D | (None, 15, 21, 64) |   |
| UpSampling2D | (None, 30, 42, 64) |   |
| Conv2DTranspose | (None, 32, 44, 64) | dropout |
| Conv2DTranspose | (None, 34, 46, 64)  | dropout |
| UpSampling2D | (None, 68, 92, 64) |   |
| Conv2DTranspose | (None, 70, 94, 32) | dropout |
| Conv2DTranspose | (None, 72, 96, 32)  | dropout |
| Conv2DTranspose | (None, 74, 98, 16) | dropout |
| UpSampling2D | (None, 148, 196, 16)  |   |
| Conv2DTranspose | (None, 150, 198, 16) |  |
| Conv2DTranspose | (None, 152, 200, 1)  |  |

3. I trained for 10 epochs with batch_size = 32, optimizer='Adam' and loss='mean_squared_error', MSE for training is 0.0040 and validation is 0.0039

![prediction](/lane_detection/fcnn/detected_lane/hist.jpg)
