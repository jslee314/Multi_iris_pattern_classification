# VGG/train_ResNet.py
from keras.utils import multi_gpu_model

from CNNModels.VGG.util.constants import *
from CNNModels.VGG.util.dataloader import DataLoader
from CNNModels.VGG.util.customcallback import CustomCallback
from CNNModels.VGG.model.vgg16v1 import VGG_16
#from PyImage.VGG.model.vgg16 import VGG_16
from CNNModels.VGG.util.output import makeoutput
import os

G = 1

os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/tensorboard')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/validationReport')

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))

print('# 1) 데이터셋 생성')
x_train, y_train, x_val, y_val, x_test, y_test, lb = DataLoader.load_data()

print('# 2) 모델 구성(add) & 엮기(compile)')
with tf.device("/cpu:0"):
    model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT,
                   depth=FLG.DEPTH, classes=len(lb.classes_))
    #model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)

# make the model parallel
model = multi_gpu_model(model, gpus=G)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()

hist = model.fit(x_train, y_train,
                 epochs=FLG.EPOCHS,
                 batch_size=FLG.BATCH_SIZE,
                 validation_data=(x_val, y_val),
                 callbacks=list_callbacks)

print('4)  모델 학습 결과 저장')
# makeoutput(x_val, y_val, model, hist, lb.classes_)

