from Multi_output_smallvggNet.util.constants import *
from Multi_output_smallvggNet.util.dataloader import DataLoader
from Multi_output_smallvggNet.util.gendataloader import ImageGenerator
from Multi_output_smallvggNet.model.irisMultioutput_vggNet import SmallerVGGNetBuilder
from Multi_output_smallvggNet.util.customcallback import CustomCallback
from Multi_output_smallvggNet.util.output import make_dir, saved_matrix, saved_model_and_graph
from Multi_output.model.irisMultioutputNet import efficientNet_output
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import sparse_categorical_crossentropy

from albumentations import (Compose,
HorizontalFlip, VerticalFlip, ShiftScaleRotate,
RandomRotate90, Transpose,  RandomSizedCrop, Flip,

RGBShift, HueSaturationValue, ChannelShuffle,
CLAHE, RandomContrast, RandomGamma, RandomBrightness,

JpegCompression, IAAPerspective, OpticalDistortion, GridDistortion, IAAAdditiveGaussianNoise, GaussNoise,
RandomBrightnessContrast,
MotionBlur, MedianBlur,
IAAPiecewiseAffine, IAASharpen, IAAEmboss, OneOf, ToFloat
)

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

defectBranch = SmallerVGGNetBuilder.vggNet_output(inputs=inputs, classes=5, finalAct="sigmoid", output_names='defect')
lacunaBranch = SmallerVGGNetBuilder.vggNet_output(inputs=inputs, classes=5, finalAct="sigmoid",  output_names='lacuna')
spokeBranch = SmallerVGGNetBuilder.vggNet_output(inputs=inputs, classes=5, finalAct="sigmoid", output_names='spoke')
spotBranch = SmallerVGGNetBuilder.vggNet_output(inputs=inputs,  classes=5, finalAct="sigmoid", output_names='spot')

print('# 2-1) 모델 구성(add)')
model = Model(inputs=inputs, outputs=[defectBranch, lacunaBranch, spokeBranch, spotBranch])
model.summary()

print('# 2-2) 엮기(compile)')
losses = {"defect": sparse_categorical_crossentropy, "lacuna": sparse_categorical_crossentropy,
          "spoke": sparse_categorical_crossentropy, "spot": sparse_categorical_crossentropy}
lossWeights = {"defect": 1.0, "lacuna": 1.0, "spoke": 1.0, "spot": 1.0}
model.compile(loss=sparse_categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'], loss_weights=lossWeights)


print('# 3) 데이터셋 생성')

AUGMENTATIONS_TRAIN  = Compose([
    HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ShiftScaleRotate(p=0.8),RandomRotate90(p=0.8), Transpose(p=0.5),
    RandomSizedCrop(min_max_height=(FLG.HEIGHT*2/3, FLG.WIDTH*2/3), height=FLG.HEIGHT, width=FLG.WIDTH,p=0.5),
    RandomContrast(p=0.5), RandomGamma(p=0.5), RandomBrightness(p=0.5)

])

AUGMENTATIONS_TEST = Compose([
    VerticalFlip(p=0.5)
])



data_dir = 'D:\Data\iris_pattern\Multi_output2_test40_train160'
dataLoader = {
    'TrainGenerator': ImageGenerator(data_dir + '/train', augmentations=AUGMENTATIONS_TRAIN ),
    'ValGenerator': ImageGenerator(data_dir + '/test', augmentations =AUGMENTATIONS_TEST)
}

train_generator = dataLoader.get('TrainGenerator')
val_generator = dataLoader.get('ValGenerator')

print('# 4) 모델 학습  fit / fit_generator(callback)')
hist = model.fit_generator(train_generator,
                           validation_data=val_generator,
                           steps_per_epoch=len(train_generator),
                           validation_steps=len(val_generator),
                           epochs=FLG.EPOCHS,
                           verbose=1)


print('# 4)  모델 학습 결과 저장')
x_val,  y_val_defect,  y_val_lacuna,  y_val_spoke,  y_val_spot, defectLB, lacunaLB, spokeLB, spotLB = DataLoader.load_data_val(data_dir)
classes = {"defect": defectLB.classes_, "lacuna": lacunaLB.classes_, "spoke": spokeLB.classes_, "spot": spokeLB.classes_}
y_val = {"defect": y_val_defect, "lacuna": y_val_lacuna, "spoke": y_val_spoke, "spot": y_val_spot}

saved_model_and_graph(model, hist, classes.keys())
saved_matrix(x_val, y_val, model, classes)




