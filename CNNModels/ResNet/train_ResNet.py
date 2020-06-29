# ResNet/train_ResNet.py

from CNNModels.ResNet.util.dataloader import DataLoader
from CNNModels.ResNet.model import ResnetBuilder
from CNNModels.ResNet.util.customcallback import CustomCallback
from CNNModels.ResNet.util.output import makeoutput
from keras import backend as K
import os

# 체크포인트 파일 저장할 디렉토리 설정
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/tensorboard')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/validationReport')

# GPU 사용하기 위한 코드
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)
)

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
# if we are using "channels first", update the input shape and channels dimension
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH, )


print('# 1) 데이터셋 생성')
x_train, y_train, x_val, y_val, x_test, y_test, lb = DataLoader.load_data()

print('# 2) 모델 구성(add) & 엮기(compile)')
model = ResnetBuilder.build_resnet_152(input_shape, len(lb.classes_))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# RMSprop

print('# 3) 모델 학습 fit(callback)')
list_callbacks = CustomCallback.callback()


hist = model.fit(x_train, y_train,
                 epochs=FLG.EPOCHS,
                 batch_size=FLG.BATCH_SIZE,
                 validation_data=(x_val, y_val),
                 callbacks=list_callbacks)

print('# 4)  모델 학습 결과 저장')
makeoutput(x_val, y_val, model, hist, lb.classes_)















