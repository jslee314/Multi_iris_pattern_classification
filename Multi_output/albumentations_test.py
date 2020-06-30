import cv2
from Multi_output.util.constants import *
from matplotlib import pyplot as plt
import random
from CNNUtil import paths
import os

from albumentations import (Compose,
HorizontalFlip, VerticalFlip, ShiftScaleRotate,
RandomRotate90, Transpose,  RandomSizedCrop, Flip,

RGBShift, HueSaturationValue, ChannelShuffle,
CLAHE, RandomContrast, RandomGamma, RandomBrightness,

JpegCompression, IAAPerspective, OpticalDistortion, GridDistortion, IAAAdditiveGaussianNoise, GaussNoise,
RandomBrightnessContrast,
MotionBlur, MedianBlur,
IAAPiecewiseAffine, IAASharpen, IAAEmboss, OneOf, ToFloat
)
# 사용불가 : ChannelShuffle, CLAHE, RandomShadow
def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst
AUGMENTATIONS = Compose([

    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=.9),
    CLAHE(p=1.0, clip_limit=2.0),
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])


# data_dir = 'D:/Data/iris_pattern/test_image/defect_lacuna_spoke_spot'
data_dir = 'D:\Data\iris_pattern\Original3\defect'

image_paths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(image_paths)


my_aug = Compose([
    HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ShiftScaleRotate(p=0.8),RandomRotate90(p=0.8), Transpose(p=0.5),
    RandomSizedCrop(min_max_height=(FLG.HEIGHT*2/3, FLG.WIDTH*2/3), height=FLG.HEIGHT, width=FLG.WIDTH,p=0.5),
    RandomContrast(p=0.5), RandomGamma(p=0.5), RandomBrightness(p=0.5)

])

for image_path in image_paths:
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))

    for num in range(3):
        # plt.subplot(211)
        # plt.imshow(image)
        aug_image = my_aug(image=image)['image']
        # plt.subplot(212)
        # plt.imshow(aug_image)
        # plt.show()
        print(image_path)
        new_path = image_path[0:-4]
        print(new_path)
        cv2.imwrite(image_path +str(num) + '_aug.png', aug_image)




#
#
# for image_path in image_paths:
#     print(image_path)
#     image = cv2.imread(image_path)
#
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = findRegion(image)
#     image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
#     plt.figure('fig1', figsize=(10, 5))
#
#     plt.subplot(241)
#     plt.imshow(image)
#
#     aug_image = RGBShift(p=1)(image=image)['image']
#     plt.subplot(242)
#     plt.imshow(aug_image)
#
#     aug_image = HueSaturationValue(p=1)(image=image)['image']
#     plt.subplot(243)
#     plt.imshow(aug_image)
#
#     aug_image = ChannelShuffle(p=1)(image=image)['image']
#     plt.subplot(244)
#     plt.imshow(aug_image)
#
#     # x
#     aug_image = CLAHE(p=1)(image=image)['image']
#     plt.subplot(245)
#     plt.imshow(aug_image)
#
#     aug_image = RandomContrast(p=1)(image=image)['image']
#     plt.subplot(246)
#     plt.imshow(aug_image)
#
#     aug_image = RandomGamma(p=1)(image=image)['image']
#     plt.subplot(247)
#     plt.imshow(aug_image)
#
#     aug_image = RandomBrightness(p=1)(image=image)['image']
#     plt.subplot(248)
#     plt.imshow(aug_image)
#
#     plt.show()
#
#
#     plt.figure('fig2', figsize=(10, 5))
#
#     plt.subplot(241)
#     plt.imshow(image)
#
#     aug_image = HorizontalFlip(p=1)(image=image)['image']
#     plt.subplot(242)
#     plt.imshow(aug_image)
#
#     aug_image = VerticalFlip(p=1)(image=image)['image']
#     plt.subplot(243)
#     plt.imshow(aug_image)
#
#     aug_image = ShiftScaleRotate(p=1)(image=image)['image']
#     plt.subplot(244)
#     plt.imshow(aug_image)
#
#     aug_image = RandomRotate90(p=1)(image=image)['image']
#     plt.subplot(245)
#     plt.imshow(aug_image)
#
#     aug_image = Transpose(p=1)(image=image)['image']
#     plt.subplot(246)
#     plt.imshow(aug_image)
#
#     aug_image = RandomSizedCrop(min_max_height=(FLG.HEIGHT*2/3, FLG.WIDTH*2/3), height=FLG.HEIGHT, width=FLG.WIDTH,p=1)(image=image)['image']
#     plt.subplot(247)
#     plt.imshow(aug_image)
#
#     aug_image = Flip(p=1)(image=image)['image']
#     plt.subplot(248)
#     plt.imshow(aug_image)
#
#     plt.show()
#
#
#
#
#     plt.figure('fig3', figsize=(10, 5))
#
#     plt.subplot(241)
#     plt.imshow(image)
#
#     aug_image = RandomBrightnessContrast(p=1)(image=image)['image']
#     plt.subplot(242)
#     plt.imshow(aug_image)
#
#     aug_image = JpegCompression(p=1)(image=image)['image']
#     plt.subplot(243)
#     plt.imshow(aug_image)
#
#     aug_image = IAAPerspective(p=1)(image=image)['image']
#     plt.subplot(244)
#     plt.imshow(aug_image)
#
#     aug_image = OpticalDistortion(p=1)(image=image)['image']
#     plt.subplot(245)
#     plt.imshow(aug_image)
#
#     aug_image = GridDistortion(p=1)(image=image)['image']
#     plt.subplot(246)
#     plt.imshow(aug_image)
#
#     aug_image = IAAAdditiveGaussianNoise(p=1)(image=image)['image']
#     plt.subplot(247)
#     plt.imshow(aug_image)
#
#     aug_image = GaussNoise(p=1)(image=image)['image']
#     plt.subplot(248)
#     plt.imshow(aug_image)
#
#     plt.show()