import numpy as np
import cv2
import random
from CNNUtil import paths
LENGTH = 180

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst


def img_padding_2(img, LENGTH):
    blank_image = np.zeros((LENGTH, LENGTH, 3), np.uint8)
    (w, h)=(img.shape[0], img.shape[1])
    len = w if w > h else h
    if len>LENGTH:
        big_img = np.zeros((len, len, 3), np.uint8)
        big_img[0:  w,  0:  h] = img
        dst = cv2.resize(big_img, (LENGTH, LENGTH))
        blank_image = dst
    else:
        blank_image[0:  w, 0:  h] = img
    return blank_image

# def img_padding(img):
#     # 이미지의 x, y가 300이 넘을 경우 작게해주기
#     blank_image = np.zeros((WIDTH, WIDTH, 3), np.uint8)
#     percent = 1
#     if(img.shape[1] >WIDTH):
#         if (img.shape[1] > img.shape[0]):  # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
#             percent = WIDTH / img.shape[1]
#         else:
#             percent = WIDTH / img.shape[0]
#     if (img.shape[0] > WIDTH):
#         if (img.shape[1] > img.shape[0]):  # 이미지의 가로가 세보다 크면 가로를 300으로 맞추고 세로를 비율에 맞춰서
#             percent = WIDTH / img.shape[1]
#         else:
#             percent = WIDTH / img.shape[0]
#
#     img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)
#
#     blank_image[ 0: img.shape[0],0: img.shape[1] ] = img
#
#     return blank_image

# data_dir = 'D:\Data\iris_pattern\Original\defect'

data_dir = 'D:/Data/iris_pattern/Binary/defect_binary/train/defect'
# data_dir = 'D:/Data/iris_pattern/test_image/11'
imagePaths = sorted(list(paths.list_images(data_dir)))
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = findRegion(image)
    image = img_padding_2(image, 180)
    # image = findRegion(image)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (HEIGHT, WIDTH))
    # cv2.namedWindow("img_re", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("img_re", 400, 400)
    cv2.imshow("img_re", image)
    # cv2.waitKey(0)