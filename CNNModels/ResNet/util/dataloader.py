'''
(x_train, x_test)
: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3)
based on the image_data_format backend setting of either channels_first or channels_last respectively.
(y_train, y_test)
: uint8 array of category labels (integers in range 0-9) with shape (num_samples,)
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
'''

from keras.utils import np_utils
from keras.datasets import cifar10
from .constants import *
import numpy as np

class DataLoader:

    @staticmethod
    def load_data():
        # 훈련셋 시험셋 로딩: train, test
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_data = np.concatenate((x_train, x_test), axis=0)
        y_data = np.concatenate((y_train, y_test), axis=0)

        # 3600 1800 600 train test val
        # 훈련셋 검증셋 분리
        x_val = x_data[54000:]
        y_val = y_data[54000:]
        x_test = x_data[36000:54000]
        y_test = y_data[36000:54000]
        x_train = x_data[0:36000]
        y_train = y_data[0:36000]

        # 3072 = 32 * 32 * 3
        x_train = x_train.reshape(36000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0
        x_test = x_test.reshape(18000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0
        x_val = x_val.reshape(6000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0

        # # 훈련셋, 검증셋 고르기
        # train_rand_idxs = np.random.choice(50000, 700)
        # val_rand_idxs = np.random.choice(10000, 300)

        # x_train = x_train[train_rand_idxs]
        # y_train = y_train[train_rand_idxs]
        # x_val = x_val[val_rand_idxs]
        # y_val = y_val[val_rand_idxs]

        print('X_train shape:', x_train.shape)
        print('x_val shape:', x_val.shape)
        print('x_test shape:', x_test.shape)
        print(x_train.shape[0], 'train samples / ', x_val.shape[0], 'val samples / ', x_test.shape[0], 'test samples')

        from sklearn.preprocessing import LabelBinarizer
        # 라벨링 전환 : 다중분류 모델일 때 -> one-hot encoding 처리
        # nb_classes = 10
        # y_train = np_utils.to_categorical(y_train, nb_classes)
        # y_val = np_utils.to_categorical(y_val, nb_classes)
        # y_test = np_utils.to_categorical(y_test, nb_classes)

        #  라벨링 전환
        '''  ['panda' 'dogs' 'cats' 'dogs' .....]  -> [ [0 1 0]\n [1 0 0 ]\n [0 0 1 ]\n .... ]  '''
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train)
        y_val= lb.transform(y_val)
        y_test= lb.transform(y_test)
        '''  lb.classes_ : ['cats' 'cogs' 'panda']  ndarray 형식  '''


        return x_train, y_train, x_val, y_val, x_test, y_test, lb


    #
    #
    # def test_data(self):
    #     # 훈련셋 시험셋 로딩: train, test
    #     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #
    #     # 훈련셋 검증셋 분리: train -> train, val
    #     x_val = x_train[40000:]
    #     y_val = y_train[40000:]
    #     x_train = x_train[0:40000]
    #     y_train = y_train[0:40000]
    #
    #     x_train = x_train.reshape(40000, 3072).astype('float32') / 255.0
    #     x_val = x_val.reshape(10000, 3072).astype('float32') / 255.0
    #     x_test = x_test.reshape(10000, 3072).astype('float32') / 255.0
    #
    #     # # 훈련셋, 검증셋 고르기
    #     # train_rand_idxs = np.random.choice(50000, 700)
    #     # val_rand_idxs = np.random.choice(10000, 300)
    #     #
    #     # x_train = x_train[train_rand_idxs]
    #     # y_train = y_train[train_rand_idxs]
    #     # x_val = x_val[val_rand_idxs]
    #     # y_val = y_val[val_rand_idxs]
    #
    #     print('X_train shape:', x_train.shape)
    #     print('x_val shape:', x_val.shape)
    #     print('x_test shape:', x_test.shape)
    #     print(x_train.shape[0], 'train samples')
    #     print(x_val.shape[0], 'test samples')
    #     print(x_test.shape[0], 'test samples')
    #
    #     # 라벨링 전환 : 원핫인코딩 (one-hot encoding) 처리
    #     nb_classes = 10
    #     y_train = np_utils.to_categorical(y_train, nb_classes)
    #     y_val = np_utils.to_categorical(y_val, nb_classes)
    #     y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    #     return x_train, y_train, x_val, y_val, x_test, y_test
    #
