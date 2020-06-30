import cv2

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,
    RandomBrightness, RandomContrast, RandomGamma,ToFloat
)
AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=1),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255)
])
aug = HorizontalFlip(p=1)

from matplotlib import pyplot as plt

image_path = 'D:\Data\iris_pattern\Multi_output2_test40_train160/train/defect_lacuna_normal_normal/defect_lacuna_normal_normal6.png'

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# image = aug(image=image)['image']
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()
