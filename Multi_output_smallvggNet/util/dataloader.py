import random
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from CNNUtil import paths
from .constants import *
import numpy as np

class DataLoader:
    kernel_sharpen = np.array(
        [[-1, -1, -1, -1, -1],
         [-1, 2, 2, 2, -1],
         [-1, 2, 8, 2, -1],
         [-1, 2, 2, 2, -1],
         [-1, -1, -1, -1, -1]]) / 8.0  # 정규화위해 8로나눔

    def img_preprocess(img):
        # cv2.imshow('orig', img)
        # cv2.waitKey(1000)

        # 3. 샤프닝
        img = cv2.filter2D(img, -1, DataLoader.kernel_sharpen)

        # 1. gray
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 히스토그램 균일화(대비조정)
        #img = cv2.equalizeHist(img)

        # 4. 리사이즈
        new_img = cv2.resize(img, (FLG.HEIGHT, FLG.WIDTH))

        # cv2.imshow('before', img)
        # cv2.waitKey(1000)
        #
        # cv2.imshow('after', new_img)
        # cv2.waitKey(1500)
        return new_img

    def load_data_test_train(data_dir):
        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images(data_dir)))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        datas = []                  # 이미지 파일을 array 형태로 변환해서 저장할 list
        defectLabels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list
        lacunaLabels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list
        spokeLabels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list
        spotLabels = []                 # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            image = DataLoader.img_preprocess(image)
            image = img_to_array(image)
            datas.append(image)
            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            (defect, lacuna, spoke, spot) = imagePath.split(os.path.sep)[-2].split("_")
            defectLabels.append(defect)
            lacunaLabels.append(lacuna)
            spokeLabels.append(spoke)
            spotLabels.append(spot)


        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        data = np.array(datas, dtype="float") / 255.0
        print("[INFO] data matrix: {} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

        defectLabels = np.array(defectLabels)
        lacunaLabels = np.array(lacunaLabels)
        spokeLabels = np.array(spokeLabels)
        spotLabels = np.array(spotLabels)

        print("[INFO] binarize both sets of labels..")
        defectLB = LabelBinarizer()
        lacunaLB = LabelBinarizer()
        spokeLB = LabelBinarizer()
        spotLB = LabelBinarizer()
        defectLabels = defectLB.fit_transform(defectLabels)
        lacunaLabels = lacunaLB.fit_transform(lacunaLabels)
        spokeLabels = spokeLB.fit_transform(spokeLabels)
        spotLabels = spotLB.fit_transform(spotLabels)

        # loop over each of the possible class labels and show them
        for (i, label) in enumerate(defectLB.classes_):
            print("{}. {}".format(i + 1, label))
            print(defectLB.classes_)
        for (i, label) in enumerate(lacunaLB.classes_):
            print("{}. {}".format(i + 1, label))
            print(lacunaLB.classes_)
        for (i, label) in enumerate(spokeLB.classes_):
            print("{}. {}".format(i + 1, label))
            print(spokeLB.classes_)
        for (i, label) in enumerate(spotLB.classes_):
            print("{}. {}".format(i + 1, label))
            print(spotLB.classes_)

        return data, defectLabels, lacunaLabels, spokeLabels, spotLabels, defectLB, lacunaLB, spokeLB, spotLB


    @staticmethod
    def load_data(data_dir):
        dir = data_dir + '/test'
        x_train, y_train_defect, y_train_lacuna, y_train_spoke, y_train_spot, defectLB, lacunaLB, spokeLB, spotLB = DataLoader.load_data_test_train(dir)
        dir = data_dir + '/train'
        x_val,  y_val_defect,  y_val_lacuna, y_val_spoke, y_val_spot, defectLB, lacunaLB, spokeLB, spotLB = DataLoader.load_data_test_train(dir)

        return x_train, x_val, y_train_defect, y_val_defect, y_train_lacuna, y_val_lacuna, y_train_spoke, y_val_spoke, y_train_spot, y_val_spot, defectLB, lacunaLB, spokeLB, spotLB

    @staticmethod
    def load_data_val(data_dir):
        dir = data_dir + '/train'
        x_val,  y_val_defect,  y_val_lacuna, y_val_spoke, y_val_spot, defectLB, lacunaLB, spokeLB, spotLB = DataLoader.load_data_test_train(dir)

        return  x_val,  y_val_defect,  y_val_lacuna,  y_val_spoke,  y_val_spot, defectLB, lacunaLB, spokeLB, spotLB
