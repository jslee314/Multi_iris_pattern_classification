'''
(x_train, x_test)
: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3)
based on the image_data_format backend setting of either channels_first or channels_last respectively.
(y_train, y_test)
: uint8 array of category labels (integers in range 0-9) with shape (num_samples,)
'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
'''

from keras.utils import np_utils
import numpy as np
from keras.datasets import cifar10
from .constants import *
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import random
from PIL import Image
import numpy
from scipy.misc import imsave

class DataLoader:

    @staticmethod
    def load_data():
        # 훈련셋 시험셋 로딩: train, test
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_data = np.concatenate((x_train, x_test), axis=0)
        y_data = np.concatenate((y_train, y_test), axis=0)

        # 4500 1000 500 train test val
        # 훈련셋 검증셋 분리
        x_train = x_data[0:45000]
        y_train = y_data[0:45000]

        x_test = x_data[45000:55000]
        y_test = y_data[45000:55000]

        x_val = x_data[55000:]
        y_val = y_data[55000:]

        # 3072 = 32 * 32 * 3
        x_train = x_train.reshape(45000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0
        x_test = x_test.reshape(10000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0
        x_val = x_val.reshape(5000, FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH).astype('float32') / 255.0

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



    @staticmethod
    def make_data_dir():
        base_dir ='D:/Data/cifar10'
        #os.mkdir(base_dir)

        # 학습, 검증, 테스트 분할을 위한 디렉토리 생성
        train_dir = os.path.join(base_dir, 'train')
        for i in range(10):
            os.makedirs(os.path.join(train_dir, str(i)))

        val_dir = os.path.join(base_dir, 'val')
        for i in range(10):
            os.makedirs(os.path.join(val_dir, str(i)))

        test_dir = os.path.join(base_dir, 'test')
        for i in range(10):
            os.makedirs(os.path.join(test_dir, str(i)))

        # bach파일에서 이미지, 레이블 파일이름 가져오기
        path = 'D:/Data/cifar-10-python/cifar-10-batches-py/'
        files = {'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch'}

        images = []
        labels = []
        filenames =[]

        for file in files:
            batch_label, image, label, filename = load_batch(path, file)
            images.extend(image)
            labels.extend(label)
            filenames.extend(filename)

        print(len(images), len(labels), len(filenames))

        for i in range(1, 60000):
            im = save_as_image(images[i])

            if i < 45000:
                nameStr = os.path.join(train_dir, str(labels[i]), filenames[i])
                imsave(nameStr, im)

            elif i < 55000:
                nameStr = os.path.join(test_dir, str(labels[i]), filenames[i])
                imsave(nameStr, im)

            else:
                nameStr = os.path.join(val_dir, str(labels[i]), filenames[i])
                imsave(nameStr, im)



    @staticmethod
    def load_data_dir():
        # 훈련셋 시험셋 로딩: train, test
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            directory='D:/Data/cifar10/train',
            target_size=(32, 32),
            batch_size=FLG.BATCH_SIZE,
            class_mode='categorical')

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            directory='D:/Data/cifar10/test',
            target_size=(32, 32),
            batch_size=FLG.BATCH_SIZE,
            class_mode='categorical')

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow_from_directory(
            directory='D:/Data/cifar10/val',
            target_size=(32, 32),
            batch_size=FLG.BATCH_SIZE,
            class_mode='categorical')

        return train_generator, test_generator, val_generator




def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_batch(path, file):
    '''
        load test_data and labels from batch file
    '''
    f = open(path + file, 'rb')
    dict = pickle.load(f, encoding='iso-8859-1')

    batch_label = dict['batch_label']
    images = dict['data']
    labels = dict['labels']
    filenames = dict['filenames']

    imagearray = np.array(images)  # (10000, 3072)
    labelarray = np.array(labels)  # (10000)
    filenamearray = np.array(filenames)  # (10000)

    return batch_label, imagearray, labelarray, filenamearray


def save_as_image(img_flat):
    '''
        Saves a data blob as an image file.
    '''
    # consecutive 1024 entries store color channels of 32x32 image
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = numpy.dstack((img_R, img_G, img_B))
    im = Image.fromarray(img)
    # im.show()

    return im
