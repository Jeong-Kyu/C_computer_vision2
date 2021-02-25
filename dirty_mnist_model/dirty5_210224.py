import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
plt.style.use('dark_background')

img_ori = cv2.imread('C:/computervision2/dirty_mnist_data/test_dirty_mnist_2nd/54999.png')
cv2.imwrite('C:/computervision2/dirty_mnist_data/test_dirty_mnist_2nd/549991.jpg',img_ori,[int(cv2.IMWRITE_JPEG_QUALITY),100])
img_ori = cv2.imread('C:/computervision2/dirty_mnist_data/test_dirty_mnist_2nd/549991.jpg')

height, width, channel = img_ori.shape
print(img_ori.shape)
plt.figure(figsize=(12,10))
plt.imshow(img_ori, cmap='gray')

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
img_thresh = cv2.adaptiveThreshold(
    img_blurred, maxValue=255.0, adaptiveMethod= cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType = cv2.THRESH_BINARY_INV, blockSize = 19, C=9
)
contours,_ = cv2.findContours(
    img_thresh, mode = cv2.RETR_LIST, method= cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype = np.uint8)
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))
plt.figure(figsize=(12,10))
plt.imshow(temp_result)

contours_dict = []

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(255,255,255),thickness=2)

    contours_dict.append({'contour':contour, 'x':x, 'y':y, 'w':w, 'h':h, 'cx':x +(w/2), 'cy':y+(h/2)})

plt.figure(figsize=(12,10))
plt.imshow(temp_result, cmap='gray')

MIN_AREA = 10
