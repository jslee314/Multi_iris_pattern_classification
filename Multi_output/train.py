import tensorflow as tf
import keras
import os
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD
from DarkNet import DarkNet53
from DataLoader import ImageNetClsTrainGenerator, ImageNetClsValGenerator
from config import flags
from ModelCheckpoint import ModelCheckpoint

loadCkptFilePath = {
    'Darknet': 'load_checkpoint/darknet_model.h5'
}
ckptFilePath = {
    'Darknet': 'checkpoint/darknet/darknet_model.h5'
}
logFilePath = {
    'Darknet': './log/darknet/darknet_log.csv'
}

dataLoader = {
    'ImageNetClsTrainGenerator': ImageNetClsTrainGenerator(img_dims=(flags.img_h, flags.img_w), batch_size=flags.batch_size),
    'ImageNetClsValGenerator': ImageNetClsValGenerator(img_dims=(flags.img_h, flags.img_w), batch_size=flags.batch_size)
}

class Trainer:
    def __init__(self, is_multi_gpu=False):
        self.is_multi_gpu = is_multi_gpu

    def train_darknet(self):
        tf.reset_default_graph()

        trainGenerator = dataLoader.get('ImageNetClsTrainGenerator')
        valGenerator = dataLoader.get('ImageNetClsValGenerator')

        if not self.is_multi_gpu:
            model, _ = DarkNet53(inputShape=(flags.img_h, flags.img_w, flags.channel)).network()

            if os.path.exists(loadCkptFilePath['Darknet']):
                print('checkpoint file :', loadCkptFilePath['Darknet'])
                model.load_weights(filepath=loadCkptFilePath['Darknet'])

            model.compile(optimizer=SGD(lr=flags.lr, momentum=0.9, decay=5e-4),
                          loss='sparse_categorical_crossentropy',
                          metrics=['acc', 'sparse_top_k_categorical_accuracy'])  # top_k_categorical_accuracy: k default 5

            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=ckptFilePath['Darknet'],
                    monitor='val_loss',
                    save_best_only=True
                ),
                keras.callbacks.CSVLogger(
                    filename=logFilePath['Darknet'],
                    append=True,
                    separator=','
                )
            ]

            model.fit_generator(
                trainGenerator,
                epochs=flags.epochs,
                steps_per_epoch=len(trainGenerator),
                validation_data=valGenerator,
                validation_steps=len(valGenerator),
                verbose=1,
                callbacks=callbacks,
                max_queue_size=64,
                workers=8,
                use_multiprocessing=True
            )
        else:
            base_model, _ = DarkNet53(inputShape=(flags.img_h, flags.img_w, flags.channel)).network()
            gpu_model = multi_gpu_model(base_model, gpus=3)

            gpu_model.compile(optimizer=SGD(lr=flags.lr, momentum=0.9, decay=5e-4),
                              loss='sparse_categorical_crossentropy',
                              metrics=['acc', 'sparse_top_k_categorical_accuracy'])  # top_k_categorical_accuracy: k default 5

            callbacks = [
                ModelCheckpoint(
                    filepath=ckptFilePath['Darknet'],
                    monitor='val_loss',
                    save_best_only=True
                ),
                keras.callbacks.CSVLogger(
                    filename=logFilePath['Darknet'],
                    append=True,
                    separator=','
                )
            ]

            gpu_model.fit_generator(
                trainGenerator,
                epochs=flags.epochs,
                steps_per_epoch=len(trainGenerator),
                validation_data=valGenerator,
                validation_steps=len(valGenerator),
                verbose=1,
                callbacks=callbacks,
                max_queue_size=64,
                workers=8,
                use_multiprocessing=True
            )


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train_darknet()