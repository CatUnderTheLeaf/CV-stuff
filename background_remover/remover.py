import cv2
import os
import numpy as np

full_path = os.path.dirname(os.path.realpath(__file__))
# image_path = full_path+"/images/PXL_20250409_101951593.jpg" #invert
# image_path = full_path+"/images/PXL_20250409_102121490.jpg"
# image_path = full_path+"/images/PXL_20250409_102136903.jpg"
# image_path = full_path+"/images/PXL_20250409_102225980.jpg" #fi
# image_path = full_path+"/images/PXL_20250409_102306293.jpg" #fi
# image_path = full_path+"/images/PXL_20250409_102319434.jpg" #fi
# image_path = full_path+"/images/PXL_20250409_102424514.jpg" #5050
# image_path = full_path+"/images/PXL_20250409_102436529.jpg"
# image_path = full_path+"/images/PXL_20250409_102458390.jpg" #invert
# image_path = full_path+"/images/PXL_20250409_102601865.jpg"
# image_path = full_path+"/images/PXL_20250409_102619587.jpg"
# image_path = full_path+"/images/PXL_20250409_102642919.jpg"#fi

image_path = full_path+"/images/white/PXL_20250409_102121490.jpg"

# image_path = full_path+"/images/white/test.jpeg"
print(image_path)


# Read the image
src = cv2.imread(image_path, 1)
image = cv2.resize(src, (480,640))
  
# Convert image to image gray
cv2.imshow('src', image) 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray) 
blur = cv2.GaussianBlur(gray, (5, 5), 0)

CANNY_THRESH_1 = 100
CANNY_THRESH_2 = 300
#-- Edge detection 
edges = cv2.Canny(blur, CANNY_THRESH_1, CANNY_THRESH_2)
# cv2.imshow('Canny', edges) 
dilate = cv2.dilate(edges, None)
# cv2.imshow('dilate', dilate) 
# erode = cv2.erode(dilate, None)
# cv2.imshow('erode', erode) 


#-- Find contours in edges, sort by area 
# contour_info = []
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# for c in contours:
#     contour_info.append((
#         c,
#         # cv2.isContourConvex(c),
#         # cv2.contourArea(c),
#     ))
# contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
# max_contour = contour_info[0]
#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape).astype(np.uint8)
# for c in contour_info:
#     cv2.fillConvexPoly(mask, c[0], (255))
for c in contours:
    cv2.fillConvexPoly(mask, c, (255))    
cv2.imshow('mask', mask)

MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,0.0,0.0) # In BGRA format
BLUR = 21
#-- Smooth mask, then blur it
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

print(mask.shape)

# mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask2', mask)
# inv_mask_stack = np.dstack([mask_inv]*4) 

#-- Blend masked img into MASK_COLOR background
mask_stack = np.dstack([mask]*4)    # Create 3-channel alpha mask
mask_stack  = mask_stack.astype('float32') / 255.0    



bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)    


img         = bgra.astype('float32') / 255.0    
masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')                    
cv2.imshow('masked', masked) 


mask4 = (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA) )* MASK_COLOR
res = bgra*(mask4/255) + mask4
cv2.imshow('res', res)                


b, g, r, _ = cv2.split(bgra)
blended_bgra = cv2.merge((b, g, r, mask))
cv2.imwrite("output.png", blended_bgra)

# weighted = cv2.addWeighted(bgra, 1, inv_mask_stack, 1, 0)
# cv2.imshow('weighted', weighted)                

other_img = cv2.bitwise_and(bgra, bgra, mask=mask)
cv2.imshow('other_img', other_img)                

# other_img2 = cv2.bitwise_and(bgra, bgra, mask=mask_inv)
# cv2.imshow('other_img2', other_img2)                

cv2.waitKey(0)
cv2.destroyAllWindows()