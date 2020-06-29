from keras.preprocessing.image import ImageDataGenerator
from .constants import *

class GendataLoader:
    def load_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            directory='D:/Data/cifar-10',
            target_size=(32, 32),
            batch_size=3,
            class_mode='categorical')
