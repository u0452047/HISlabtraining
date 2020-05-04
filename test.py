import numpy as np
import cv2
from matplotlib import pyplot as plt

# 使用 OpenCV 讀取圖檔
img_bgr = cv2.imread('image.jpg')

# 將 BGR 圖片轉為 RGB 圖片
img_rgb = img_bgr[:,:,::-1]

# 或是這樣亦可
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# 使用 OpenCV 讀取灰階圖檔

# 使用 Matplotlib 顯示圖片
plt.imshow(img_rgb)
plt.show()

img_gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 使用 Matplotlib 顯示圖片
plt.imshow(img_gray, cmap = 'gray')
plt.show()