# VGG/train_ResNet.py
from CNNModels.VGG.util.constants import *

from CNNModels.VGG.util.dataloader import DataLoader
from CNNModels.VGG.util.customcallback import CustomCallback
from CNNModels.VGG.model.vgg16v1 import VGG_16
#from PyImage.VGG.model.vgg16 import VGG_16
from CNNModels.VGG.util. output import makeoutput
import os
import numpy as np

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/tensorboard')
os.makedirs('./output/'+FLG.PROJECT_NAME+'/validationReport')

# ensure_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt')
# ensure_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
# ensure_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
# ensure_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
# ensure_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

print(' -- 1. 데이터셋 생성 -- ')
# DataLoader.make_data_dir()
train_generator, test_generator, val_generator = DataLoader.load_data_dir()


print(' -- 2. 모델 구성(add) & 엮기(compile) -- ')
model = VGG_16(width=FLG.WIDTH, height=FLG.HEIGHT, depth=FLG.DEPTH, classes=10)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('steps_per_epoch' + str(train_generator.samples // train_generator.batch_size))   # 2249
print('validation_steps' + str(val_generator.samples // train_generator.batch_size))        # 250

print('-- 3. 모델 학습 fit(callback) --')
list_callbacks = CustomCallback.callback()
hist = model.fit_generator(generator=train_generator,
                           steps_per_epoch=train_generator.samples // train_generator.batch_size,
                           epochs=FLG.EPOCHS,
                           validation_data=val_generator,
                           validation_steps=val_generator.samples // train_generator.batch_size,
                           callbacks=list_callbacks)

print("-- 4. 모델 평가하기(Evaluate) --")
''' 테스트 입력과 출력을 모두 사용합니다.
 먼저 훈련 입력을 사용하여 출력을 예측 한 다음 테스트 출력과 비교하여 성능을 평가합니다.
  따라서 성능의 척도, 즉 정확도를 제공 '''
scores = model.evaluate_generator( test_generator,
                                   steps=test_generator.samples // test_generator.batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


print("-- 4. 모델 예측하기(Predict) --")
''' 테스트 데이터를 가져와 출력을 제공 '''
predictions = model.predict_generator( test_generator,
                                       steps=test_generator.samples//test_generator.batch_size, verbose=0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(predictions)



print('  - 5.  모델 학습 결과 저장 -- ')
makeoutput(test_generator, predictions, model, hist, classes=10)














