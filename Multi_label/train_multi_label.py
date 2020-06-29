from Multi_label.util.constants import *
from Multi_label.util.dataloader import DataLoader
from Multi_label.util.customcallback import CustomCallback
from keras import backend as K
from Multi_label.util.output import makeoutput, make_dir
from CNNModels.VGG.model.vgg16v1 import VGG_16
from CNNModels.VGG.model.smallvggnet import SmallVGGNet
from CNNModels.EfficientNet .efficientnet import efficientNet_factory


make_dir('./output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
make_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

# config = tf.ConfigProto(
#     gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))


input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

print('# 1) 데이터셋 생성')
# DataSet 1 : 옷의 색과 종류를 구분하는 데이터
#x_train, y_train, x_val, y_val, lb = DataLoader.load_data_one()

# # DataSet 2 :  4가지의 물체를 구분하는 데이터
x_train, y_train, x_val, y_val, lb = DataLoader.load_data_two()

# # DataSet 3 :  홍채의 패턴을 구분하는 데이터 (AB: absorptionting/CH: cholesterolring/ ST : stressring)
# x_train, y_train, x_val, y_val, lb = DataLoader.load_data_three()

from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


print('# 2) 모델 구성(add) & 엮기(compile)')
print('The number of class : ' + str(len(lb.classes_)))
# model = MobileNetBuilder.build_mobilenet_v2(input_shape, len(lb.classes_))
#model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=len(lb.classes_))
#model = SmallVGGNet.build( width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=len(lb.classes_))
model, model_size = efficientNet_factory('efficientnet-b1',  load_weights=None, input_shape=(FLG.WIDTH, FLG.HEIGHT, FLG.DEPTH), classes=len(lb.classes_))
model.summary()

print('#  엮기(compile)')
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])


print('# 3) 모델 학습 fit(callback)')
# list_callbacks = CustomCallback.callback()

print('#  generator(fit_generator)')
hist = model.fit(x_train, y_train,
                 epochs=FLG.EPOCHS,
                 batch_size=FLG.BATCH_SIZE,
                 validation_data=(x_val, y_val))
                 # callbacks=list_callbacks)

# 클래스 당 1,000 개 미만의 이미지로 작업하는 경우 데이터 확대는 가장 좋은 방법이며 "필수"
# hist = model.fit_generator(
# 	aug.flow(x_train, y_train, batch_size=FLG.BATCH_SIZE),
# 	validation_data=(x_val, y_val),
# 	steps_per_epoch=len(x_train) // FLG.BATCH_SIZE,
# 	epochs=FLG.EPOCHS, verbose=1)


print('# 4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)