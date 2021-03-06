# ResNet/train_ResNet.py
"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset_2.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from CNNModels.ResNet.util.constants import *
from CNNModels.ResNet.util.dataloader import DataLoader
from CNNModels.ResNet.model.resnet import ResnetBuilder
from CNNModels.ResNet.util.customcallback import CustomCallback
from CNNModels.ResNet.util.output import makeoutput
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import os
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/tensorboard')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/validationReport')

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)
)

data_augmentation = True

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
# if we are using "channels first", update the input shape and channels dimension
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH, )


print('# 1) 데이터셋 생성')
x_train, y_train, x_val, y_val, x_test, y_test, lb = DataLoader.load_data()

print('# 2) 모델 구성(add) & 엮기(compile)')
model = ResnetBuilder.build_resnet_18(input_shape, len(lb.classes_))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()

if not data_augmentation:
    print('data augmentation 사용 안함')
    hist = model.fit(x_train, y_train,
                     epochs=FLG.EPOCHS,
                     batch_size=FLG.BATCH_SIZE,
                     validation_data=(x_val, y_val),
                     callbacks=list_callbacks)
else:
    print('real-time data augmentation 사용 함')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,           # set input mean to 0 over the dataset_2
        samplewise_center=False,            # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset_2
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=FLG.BATCH_SIZE),
                        steps_per_epoch=x_train.shape[0] // FLG.BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        epochs=FLG.EPOCHS, verbose=1, max_q_size=100,
                        callbacks=list_callbacks)

print('4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)