from Multi_output.util.constants import *
from Multi_output.util.dataloader import DataLoader
from Multi_output.util.gendataloader import ImageGenerator
from Multi_output.util.customcallback import CustomCallback
from Multi_output.util.output import make_dir, saved_matrix, saved_model_and_graph
from Multi_output.model.irisMultioutputNet import efficientNet_output
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import sparse_categorical_crossentropy

input_shape = (FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH)
if K.image_data_format() == "channels_first":
    input_shape = (FLG.DEPTH, FLG.HEIGHT, FLG.WIDTH)

print('# 0) 저장할 파일 생성')
make_dir('./output/'+FLG.PROJECT_NAME +'/modelsaved/ckpt')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/ckpt_pb')
make_dir('./output/'+FLG.PROJECT_NAME+'/modelsaved/h5')
make_dir('./output/'+FLG.PROJECT_NAME+'/tensorboard')
make_dir('./output/'+FLG.PROJECT_NAME+'/validationReport')

print('# 1) 모델 구성(add) & 엮기(compile)')
inputs = Input(shape=(FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH))
defectBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='defect')
lacunaBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='lacuna')
spokeBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='spoke')
spotBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='spot')

print('# 2-1) 모델 구성(add)')
model = Model(inputs=inputs, outputs=[defectBranch, lacunaBranch, spokeBranch, spotBranch])
model.summary()

print('# 2-2) 엮기(compile)')
losses = {"defect": sparse_categorical_crossentropy, "lacuna": sparse_categorical_crossentropy,
          "spoke": sparse_categorical_crossentropy, "spot": sparse_categorical_crossentropy}
lossWeights = {"defect": 1.0, "lacuna": 1.0, "spoke": 1.0, "spot": 1.0}
model.compile(loss=sparse_categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'], loss_weights=lossWeights)


print('# 3) 데이터셋 생성')
data_dir = 'D:\Data\iris_pattern\Multi_output2_test40_train160'
x_train, x_val, y_train_defect, y_val_defect, y_train_lacuna, y_val_lacuna, y_train_spoke, y_val_spoke, y_train_spot, y_val_spot, defectLB, lacunaLB, spokeLB, spotLB = DataLoader.load_data(data_dir)

train_datagen = ImageDataGenerator(rotation_range=180, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.1, zoom_range=0.2,
	horizontal_flip=True, vertical_flip = True, fill_mode="nearest")

test_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip = True)

def multiple_outputs(datagen, x,  y_defect, y_lacuna, y_spoke, y_spot):
    genY1 = datagen.flow(x, y_defect, batch_size=FLG.BATCH_SIZE, seed=5)
    genY2 = datagen.flow(x, y_lacuna, batch_size=FLG.BATCH_SIZE, seed=5)
    genY3 = datagen.flow(x, y_spoke, batch_size=FLG.BATCH_SIZE, seed=5)
    genY4 = datagen.flow(x, y_spot, batch_size=FLG.BATCH_SIZE, seed=5)

    while True:
        y1 = genY1.next()
        y2 = genY2.next()
        y3 = genY3.next()
        y4 = genY4.next()
        # return image batch and 4 sets of lables
        yield y1[0], [y1[1], y2[1], y3[1], y4[1]]


train_generator = multiple_outputs(train_datagen, x_train,  y_train_defect, y_train_lacuna, y_train_spoke, y_train_spot)
val_generator = multiple_outputs(test_datagen, x_val,  y_val_defect, y_val_lacuna, y_val_spoke, y_val_spot)

print('# 4) 모델 학습  fit / fit_generator(callback)')
# list_callbacks = CustomCallback.callback()
y_train = {"defect": y_train_defect, "lacuna": y_train_lacuna, "spoke": y_train_spoke, "spot": y_train_spot}
y_val = {"defect": y_val_defect, "lacuna": y_val_lacuna, "spoke": y_val_spoke, "spot": y_val_spot}

# hist = model.fit(x_train, y_train,
#                  epochs=FLG.EPOCHS,
#                  batch_size=FLG.BATCH_SIZE,
#                  validation_data=(x_val,  y_val))
#                  # callbacks=list_callbacks)
hist = model.fit_generator(train_generator,
                           validation_data=val_generator,
                           steps_per_epoch=len(x_train) // FLG.BATCH_SIZE,
                           validation_steps=len(x_val) // FLG.BATCH_SIZE,
                           epochs=FLG.EPOCHS,
                           verbose=1)

classes = {"defect": defectLB.classes_, "lacuna": lacunaLB.classes_, "spoke": spokeLB.classes_, "spot": spokeLB.classes_}
print('# 4)  모델 학습 결과 저장')
# show_graph(hist)
saved_model_and_graph(model, hist, classes.keys())
saved_matrix(x_val, y_val, model, classes)