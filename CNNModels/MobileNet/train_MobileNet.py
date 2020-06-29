from CNNModels.MobileNet.util.dataloader import DataLoader
from CNNModels.MobileNet.util.customcallback import CustomCallback
from CNNModels.MobileNet.util.output import makeoutput, make_dir
from CNNModels.MobileNet.model.mobilenet import MobileNetBuilder
from CNNModels.MobileNet.model.mobilenet_v3 import MobileNetV3
from CNNModels.MobileNet.util.constants import *
from tensorflow.keras import backend as K

make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
make_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH, )


print('# 1) 데이터셋 생성')
x_train, y_train, x_val, y_val, x_test, y_test, lb = DataLoader.load_data()


print('# 2) 모델 구성(add) & 엮기(compile)')
model = MobileNetBuilder.build_mobilenet_v1(input_shape=input_shape, classes=len(lb.classes_))
model = MobileNetBuilder.build_mobilenet_v2(input_shape=input_shape, classes=len(lb.classes_))
#model = MobileNetV3(multiplier=1.0, input_shape=input_shape, num_classes=len(lb.classes_)).build(config='large')

print('#  엮기(compile)')
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()

hist = model.fit(x_train, y_train,
                 epochs=FLG.EPOCHS,
                 batch_size=FLG.BATCH_SIZE,
                 validation_data=(x_val, y_val))
                 # callbacks=list_callbacks)

print('# 4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)