import random
import cv2
import os
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from CNNUtil import paths
from .constants import *

import numpy as np

class DataLoader:

    @staticmethod
    def load_data_one():

        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images("dataset_2")))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        datas = []                # 이미지 파일을 array 형태로 변환해서 저장할 list
        labels = []             # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            # cv2.imshow('before', image)
            # cv2.waitKey(1000)

            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
            # cv2.imshow('after', image)
            # cv2.waitKey(1500)
            image = img_to_array(image)
            datas.append(image)

            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            label = imagePath.split(os.path.sep)[-2].split("_")
            labels.append(label)

        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        data = np.array(datas, dtype="float") / 255.0
        labels = np.array(labels)

        # print("     [INFO] data matrix: {} images ({:.2f}MB)".format( len(imagePaths), data.nbytes / (1024 * 1000.0)))

        # scikit-learn의  multi-label binarizer 을 사용하여 레이블을 이진화
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)      # [blue, jeans] -> [0 1 1 0 0 0 ]

        # 각 레이블이 무엇을 의미하는지 출력
        for (i, label) in enumerate(mlb.classes_):
            print("     {}. {}".format(i + 1, label))

        #  훈련 용 데이터의 80 %와 시험용 나머지 20 %를 사용하여 데이터를 훈련 및 테스트 분할로 분할
        (x_train, x_val, y_train, y_val) = train_test_split(data, labels,
                                                            test_size=0.2, random_state=42)

        return x_train, y_train, x_val, y_val, mlb

    @staticmethod
    def load_data_two():
        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images('D:\Data\iris_pattern\Multi-label_3st')))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        datas = []                # 이미지 파일을 array 형태로 변환해서 저장할 list
        labels = []             # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            # cv2.imshow('before', image)
            # cv2.waitKey(1000)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
            # cv2.imshow('after', image)
            # cv2.waitKey(1500)
            image = img_to_array(image)
            datas.append(image)

            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            label = imagePath.split(os.path.sep)[-2].split("_")
            labels.append(label)

        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        data = np.array(datas, dtype="float") / 255.0
        labels = np.array(labels)

        print("[INFO] data matrix: {} images ({:.2f}MB)".format( len(imagePaths), data.nbytes / (1024 * 1000.0)))

        # scikit-learn의  multi-label binarizer 을 사용하여 레이블을 이진화
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)      # [blue, jeans] -> [0 1 1 0 0 0 ]

        # 각 레이블이 무엇을 의미하는지 출력
        for (i, label) in enumerate(mlb.classes_):
            print("     {}. {}".format(i + 1, label))

        #  훈련 용 데이터의 80 %와 시험용 나머지 20 %를 사용하여 데이터를 훈련 및 테스트 분할로 분할
        (x_train, x_val, y_train, y_val) = train_test_split(data, labels,
                                                            test_size=0.2, random_state=42)

        return x_train, y_train, x_val, y_val, mlb

    @staticmethod
    def load_data_three():
        print("[INFO] 학습할 이미지 로드 (로드시 경로들 무작위로 섞음)")
        imagePaths = sorted(list(paths.list_images("dataset_iris")))
        random.seed(42)
        random.shuffle(imagePaths)

        # data()와 labels 초기화
        data = []                # 이미지 파일을 array 형태로 변환해서 저장할 list
        labels = []             # 해당 이미지의 정답(label) 세트를 저장할 list

        print("[INFO] 모든 이미지에 대하여 이미지 데이터와 라벨을 추출 ..")
        for imagePath in imagePaths:
            # a. 이미지를 로드하고, 전처리를 한 후 데이터 목록에 저장
            image = cv2.imread(imagePath)
            # 전처리 1) resize
            image = cv2.resize(image, (FLG.HEIGHT, FLG.WIDTH))
            # 전처리 2) to gray
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = img_to_array(image)
            data.append(image)

            # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
            label = imagePath.split(os.path.sep)[-2].split("_")
            labels.append(label)

        print("[INFO] scale the raw pixel 의 밝기값을 [0, 1]으로 조정하고 np.array로 변경")
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        # print("     [INFO] data matrix: {} images ({:.2f}MB)".format( len(imagePaths), data.nbytes / (1024 * 1000.0)))

        # scikit-learn의  multi-label binarizer 을 사용하여 레이블을 이진화
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)      # [blue, jeans] -> [0 1 1 0 0 0 ]

        # 각 레이블이 무엇을 의미하는지 출력
        for (i, label) in enumerate(mlb.classes_):
            print("     {}. {}".format(i + 1, label))

        #  훈련 용 데이터의 80 %와 시험용 나머지 20 %를 사용하여 데이터를 훈련 및 테스트 분할로 분할
        (x_train, x_val, y_train, y_val) = train_test_split(data, labels,
                                                            test_size=0.2, random_state=42)

        return x_train, y_train, x_val, y_val, mlb
