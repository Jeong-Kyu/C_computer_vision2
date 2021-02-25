

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

'''
# 이미지 불러오기
Img = cv2.imread('../dacon12/data/train/00000.PNG', cv2.IMREAD_GRAYSCALE)
# 이미지 전처리 -------------------------------------
#254보다 작고 0이아니면 0으로 만들어주기
img2 = np.where((Img <= 254) & (Img != 0), 0, Img)
# 이미지 팽창
img2 = cv2.dilate(img2, kernel=np.ones((2, 2), np.uint8), iterations=1)
# 블러 적용, 노이즈 제거
img2 = cv2.medianBlur(src=img2, ksize= 5)
# canny
img2 = cv2.Canny(img2, 30, 70)
print(img2.shape)
cv2.imwrite('../dacon12/data/newtrain/00000.jpg', img2)
'''

# 50000개 저장용
number = 50000

for a in np.arange(0,number):
    Img = cv2.imread('C:/computervision2_data/dirty_mnist_data/dirty_mnist_2nd/' + str(a).zfill(5) + '.png', cv2.IMREAD_GRAYSCALE)
    #254보다 작고 0이아니면 0으로 만들어주기
    img2 = np.where((Img <= 254) & (Img != 0), 0, Img)
    # 이미지 팽창
    img2 = cv2.dilate(img2, kernel=np.ones((2, 2), np.uint8), iterations=1)
    # 블러 적용, 노이즈 제거
    img2 = cv2.medianBlur(src=img2, ksize= 5)
    # canny
    # img2 = cv2.Canny(img2, 30, 70)

    cv2.imwrite('C:/computervision2_data/dirty_mnist_data/clean_dirty/train/'+ str(a).zfill(5) + '.png', img2)

print('done')