import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

full_path = os.path.realpath(__file__)
# Load the left and right images
img_left_color = cv2.imread(os.path.dirname(full_path)+'/images/left.png')
right_img = cv2.imread(os.path.dirname(full_path)+'/images/right.png', 0)
left_img = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2GRAY)

# Blur images to smooth the noise
left_img = cv2.blur(left_img,(5,5))
right_img = cv2.blur(right_img,(5,5))
cv2.imwrite(os.path.dirname(full_path)+'/images/left_blur.png',left_img)
cv2.imwrite(os.path.dirname(full_path)+'/images/right_blur.png',right_img)
disparity_map = np.empty_like(left_img)

# camera matrices and other parameters
cam0 = np.array(
[[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
[0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
[0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]]).astype('float32')
cam1 = np.array(
[[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02],
[0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02],
[0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]]).astype('float32')
baseline=540
vmin=2#10
vmax=78
block = 15 #11

#############################################################
# disparity map creation
stereo = cv2.StereoSGBM_create(numDisparities=vmax, blockSize=block,minDisparity=vmin)

title_window = 'Disparity map'
def compute_and_show_image():
    global disparity_map
    # disparity_map = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity_map = stereo.compute(left_img, right_img)
    disparity_color = cv2.applyColorMap(np.array(disparity_map.astype(np.float32) / 16.0, dtype = np.uint8), cv2.COLORMAP_JET)

    cv2.imshow(title_window, disparity_color)
    cv2.imwrite(os.path.dirname(full_path)+'/images/disparity_color.png',disparity_color)
#############################################################
# To view a window with trackbars

# minDisparity = vmin
# numDisparities = vmax
# block_size = block

# def on_num(val):
#     stereo.setNumDisparities(val)
#     compute_and_show_image()

# def on_min(val):
#     stereo.setMinDisparity(val)
#     compute_and_show_image()
    
# def on_block(val):
#     stereo.setBlockSize(val)
#     compute_and_show_image()


# cv2.namedWindow(title_window)

# cv2.createTrackbar('numDisparities', title_window , 0, numDisparities, on_num)
# cv2.createTrackbar('minDisparity', title_window , 0, minDisparity, on_min)
# cv2.createTrackbar('block_size', title_window , 0, block_size, on_block)

# # Set default value for Max HSV trackbars
# cv2.setTrackbarPos('numDisparities', title_window, 79)
# cv2.setTrackbarPos('minDisparity', title_window, 1)
# cv2.setTrackbarPos('block_size', title_window, 15)

# # Wait until user press some key
# cv2.waitKey()

#############################################################

compute_and_show_image()

plt.imshow(disparity_map, 'rainbow')
plt.show()
plt.imsave(os.path.dirname(full_path)+'/images/disparity_map.png',disparity_map)

# Calculate matrix Q - disparity-to-depth mapping matrix
rev_proj_matrix = np.zeros((4,4)) # to store the output
cv2.stereoRectify(cameraMatrix1 = cam0,cameraMatrix2 = cam1,
                  distCoeffs1 = 0, distCoeffs2 = 0,
                  imageSize = left_img.shape[:2],
                  R = np.identity(3), T = np.array([baseline/1000, 0., 0.]),
                  R1 = None, R2 = None,
                  P1 =  None, P2 =  None, 
                  Q = rev_proj_matrix)

# project depth image points to 3d 
points = cv2.reprojectImageTo3D(disparity_map, rev_proj_matrix)

#reflect on x axis
reflect_matrix = np.identity(3)
reflect_matrix[0] *= -1
points = np.matmul(points,reflect_matrix)
points = points[disparity_map > disparity_map.min()]

#extract colors from image
colors = cv2.cvtColor(img_left_color, cv2.COLOR_BGR2RGB)
out_colors = colors[disparity_map > disparity_map.min()]

def write_ply(fn, points, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = np.array(colors).reshape(-1,3)
    points = points.reshape(-1, 3)
    verts = np.hstack([points, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# save as .ply file
write_ply(os.path.dirname(full_path)+'/images/out.ply', points, out_colors)

# visulize in 3d
pcd = o3d.io.read_point_cloud(os.path.dirname(full_path)+'/images/out.ply')
o3d.visualization.draw_geometries([pcd])

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()