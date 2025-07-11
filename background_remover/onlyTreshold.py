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
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Binary threshold (you can tweak this)
# _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

# Invert if needed (foreground should be white)
mask = cv2.bitwise_not(mask)

# Optional: clean up mask
# mask = cv2.dilate(mask, None, iterations=2)
# mask = cv2.erode(mask, None, iterations=2)

# Convert single-channel mask to 3-channel
mask_3ch = cv2.merge([mask, mask, mask])

# Apply mask to original image
result = cv2.bitwise_and(image, mask_3ch)

# Optional: add alpha channel
alpha = mask
b, g, r = cv2.split(image)
rgba = cv2.merge([b, g, r, alpha])

# Show result
cv2.imshow("Masked Image", result)

cv2.imshow('mask', mask) 


cv2.waitKey(0)
cv2.destroyAllWindows()