import cv2
import numpy as np
import random
from CNNUtil import paths

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]



    cv2.imshow("threshold", thr)
    cv2.imshow("dst", dst)
    cv2.rectangle(img, (x, y), (x+len, y+len), (0, 0, 255), 3)
    cv2.imshow("retangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imagePath = 'D:/Data/iris_pattern/test_image/test/lung_20191025_21359d2a-2182-4e19-84a1-a11ae064c91a.png'
image = cv2.imread(imagePath)
findRegion(image)

# data_dir = 'D:/Data/iris_pattern/test_image/test'
# imagePaths = sorted(list(paths.list_images(data_dir)))
# random.seed(42)
# random.shuffle(imagePaths)
#
# for imagePath in imagePaths:
#     print(imagePath)
#     image = cv2.imread(imagePath)
#     findRegion(image)