import os
import numpy as np
import cv2
import re
import xml.etree.ElementTree as elemTree
from keras.utils import Sequence

class ImageNetClsTrainGenerator(Sequence):

    def __init__(self, img_dims=(224, 224), batch_size=32):
        self.img_path = 'G:/04.dataset/03.imagenet/imagenet/imagenet_2012/Data/CLS-LOC/train'
        self.label_path = 'G:/04.dataset/03.imagenet/imagenet/imagenet_2012/imagenet_label/imagenet_labels.txt'
        self.img_paths, self.imagenet_labels = self.getImagePaths()
        self.img_dims = img_dims
        self.batch_size = batch_size
        self.indices = np.random.permutation(len(self.img_paths))

    def getImagePaths(self):
        img_paths, imagenet_labels = [], dict()

        with open(self.label_path, 'r') as f:
            for line in f:
                key, value = line.split()[0:2]
                imagenet_labels[key] = value

        for (path, dirs, files) in os.walk(self.img_path):
            for file in files:
                img_paths.append(os.path.join(path, file))

        return img_paths, imagenet_labels

    def load_img(self, img_path, img_dims):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_dims[1], img_dims[0]), cv2.INTER_AREA)

        return img

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx+1) * self.batch_size]

        img_paths = [self.img_paths[i] for i in batch_idx]

        x_batch, y_batch = [], []

        for img_path in img_paths:
            x_batch.append(self.load_img(img_path, self.img_dims))
            y_batch.append(int(self.imagenet_labels.get(re.search('.*\\\\(.*)\\\\.*.JPEG', img_path).group(1)))-1)

        x_batch = (np.array(x_batch) / 255.).reshape(-1, self.img_dims[0], self.img_dims[1], 3)
        y_batch = np.array(y_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.random.permutation(len(self.img_paths))


class ImageNetClsValGenerator(Sequence):

    def __init__(self, img_dims=(224, 224), batch_size=32):
        self.img_path = 'G:/04.dataset/03.imagenet/imagenet/imagenet_2012/Data/CLS-LOC/val'
        self.label_path = 'G:/04.dataset/03.imagenet/imagenet/imagenet_2012/imagenet_label/imagenet_labels.txt'
        self.xml_path = 'G:/04.dataset/03.imagenet/imagenet/imagenet_2012/Annotations/CLS-LOC/val'
        self.img_paths, self.imagenet_labels = self.getImagePaths()
        self.img_dims = img_dims
        self.batch_size = batch_size
        self.indices = np.random.permutation(len(self.img_paths))

    def getImagePaths(self):
        img_paths, imagenet_labels = [], dict()

        with open(self.label_path, 'r') as f:
            for line in f:
                key, value = line.split()[0:2]
                imagenet_labels[key] = value

        for (path, dirs, files) in os.walk(self.img_path):
            for file in files:
                img_paths.append(os.path.join(path, file))

        return img_paths, imagenet_labels

    def load_img(self, img_path, img_dims):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_dims[1], img_dims[0]), cv2.INTER_AREA)

        return img

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size: (idx+1) * self.batch_size]

        img_paths = [self.img_paths[i] for i in batch_idx]

        x_batch, y_batch = [], []

        for img_path in img_paths:
            x_batch.append(self.load_img(img_path, self.img_dims))
            fname = re.search('.*\\\\(.*).JPEG', img_path).group(1) + '.xml'
            tree = elemTree.parse(os.path.join(self.xml_path, fname))
            object = tree.find('object')
            y_batch.append(int(self.imagenet_labels.get(object.find('name').text)) -1)

        x_batch = (np.array(x_batch) / 255.).reshape(-1, self.img_dims[0], self.img_dims[1], 3)
        y_batch = np.array(y_batch)

        return x_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.random.permutation(len(self.img_paths))