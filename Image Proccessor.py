# Modules
from PIL import Image
import cv2
import os

#Variables
rows, cols = 64, 64
crop = 0.0

# Input Image
img_name = 'bloodymary.jpeg'
img = cv2.imread(img_name) # cv2 import as BGR np array
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

'''
# Crop outer edges
w, h = img.shape[1], img.shape[0]
img = img[int(h*crop):h-int(h*crop), int(w*crop):w-int(w*crop)]


# Crop to square
w, h = img.shape[1], img.shape[0]
if w > h : # Crop to square
    img = img[0:h, int((w-h)/2):w-int((w-h)/2)]
elif h > w :
    img = img[int((h - w) / 2):h - int((h - w) / 2), 0:w]
'''

# Add padding to make square
w, h = img.shape[1], img.shape[0]
if w > h : # Crop to square
    img = cv2.copyMakeBorder(img, int((w-h)/2), int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
elif h > w :
    img = cv2.copyMakeBorder(img, 0, 0, int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT, value=(0,0,0))
# Rescale to target size
img = cv2.resize(img, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)


# Display
img = Image.fromarray(img, 'RGB')
img.save('paddedimg.png', 'PNG')

