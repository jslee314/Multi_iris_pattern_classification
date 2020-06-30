from keras.preprocessing.image import ImageDataGenerator
from .constants import *
from tensorflow.python.keras.utils import Sequence
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
import cv2
import os
import numpy as np
from CNNUtil import paths


class ImageGenerator(Sequence):

    def __init__(self, data_dir= 'D:\Data\iris_pattern\Multi_output2_test40_train160', augmentations=None):
        self.total_paths, self.defect_lbs, self.lacuna_lbs, self.spoke_lbs, self.spot_lbs = self.get_total_data_path(data_dir)
        self.batch_size = FLG.BATCH_SIZE
        self.indices = np.random.permutation(len(self.total_paths))
        self.augment = augmentations

    def get_total_data_path(self, data_dir):
        total_paths, defect_lbs, lacuna_lbs, spoke_lbs, spot_lbs = [], [], [], [], []  # 이미지 path와 정답(label) 세트를 저장할 list

        image_paths = sorted(list(paths.list_images(data_dir)))
        for image_path in image_paths:
            # a. 이미지 전체 파일 path 저장
            total_paths.append(image_path)
            # b. 이미지 파일 path에서  이미지의 정답(label) 세트 추출
            (defect, lacuna, spoke, spot) = image_path.split(os.path.sep)[-2].split("_")
            defect_lbs.append(defect)
            lacuna_lbs.append(lacuna)
            spoke_lbs.append(spoke)
            spot_lbs.append(spot)

        defect_lbs = np.array(defect_lbs)
        lacuna_lbs = np.array(lacuna_lbs)
        spoke_lbs = np.array(spoke_lbs)
        spot_lbs = np.array(spot_lbs)

        defect_lbs = LabelBinarizer().fit_transform(defect_lbs)
        lacuna_lbs = LabelBinarizer().fit_transform(lacuna_lbs)
        spoke_lbs = LabelBinarizer().fit_transform(spoke_lbs)
        spot_lbs = LabelBinarizer().fit_transform(spot_lbs)

        return total_paths, defect_lbs, lacuna_lbs, spoke_lbs, spot_lbs

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
        if self.augment is not None:
            image = self.augment(image=image)['image']
        image = img_to_array(image)
        return image

    def __len__(self):
        return len(self.total_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        x_batch, defect_batch, lacuna_batch, spoke_batch, spot_batch = [], [], [], [], []

        img_paths = [self.total_paths[i] for i in batch_idx]
        defect_batch = [self.defect_lbs[i] for i in batch_idx]
        lacuna_batch = [self.lacuna_lbs[i] for i in batch_idx]
        spoke_batch = [self.spoke_lbs[i] for i in batch_idx]
        spot_batch = [self.spot_lbs[i] for i in batch_idx]

        for img_path in img_paths:
            x_batch.append(self.load_image(img_path))

        x_batch = np.array(x_batch, dtype="float") / 255.0


        return [x_batch], [defect_batch, lacuna_batch, spoke_batch, spot_batch]

    def on_epoch_end(self):
        print(self.batch_size)
        self.indices = np.random.permutation(len(self.total_paths))

