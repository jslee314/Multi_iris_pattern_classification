from .util.constants import *
from .util.dataloader import DataLoader
from numpy import argmax
import keras
import numpy as np

# 1. 데이터셋 생성
train_generator, test_generator, val_generator = DataLoader.load_data_dir()

# 2. modelsaved 불러오기
model = keras.models.load_model(FLG.H5 + '.h5')

# 3. 예측 (test)
yhat = model.predict_classes(x_test)

for i in range(5):
    print('True : '      + str(argmax(y_test[i])) )
    print( 'Predict : ' + str(yhat[i]))


    # 모델 예측하기
    print("-- Predict --")
    predictions = model.predict_generator(
        test_generator,
        steps=5)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(predictions)


print("-- 4. 모델 예측하기(Predict) --")
predictions = model.predict_generator( test_generator,
    steps=test_generator.samples // test_generator.batch_size)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(predictions)

