from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from CNNUtil.imutils import imutils
import cv2
from IrisPattern.util.dataloader import DataLoader
from CNNModels.EfficientNet .efficientnet import efficientNet_factory
from IrisPattern.util.constants import *
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.layers import Input
from Multi_output.model.irisMultioutputNet import efficientNet_output
from tensorflow.keras.models import Model
import random
from CNNUtil import paths
import os

def findRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rct, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    len = w if w > h else h
    dst = img[y : y+len, x: x+len]
    return dst

print(' [STEP 2] 모델 불러오기')
inputs = Input(shape=(FLG.HEIGHT, FLG.WIDTH, FLG.DEPTH))
defectBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='defect')
lacunaBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='lacuna')
spokeBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='spoke')
spotBranch = efficientNet_output('efficientnet-b1', load_weights=None, input_tensor=inputs, classes=2, output_names='spot')

model = Model(inputs=inputs, outputs=[defectBranch, lacunaBranch, spokeBranch, spotBranch])

h5_weights_path = './model_weights.h5'
# h5_weights_path ='./output/iris_pattern_sharpen_300_32/modelsaved/h5/model_weights.h5'

model.load_weights(h5_weights_path)
losses = {"defect": sparse_categorical_crossentropy, "lacuna": sparse_categorical_crossentropy, "spoke": sparse_categorical_crossentropy, "spot": sparse_categorical_crossentropy}
lossWeights = {"defect": 1.0, "lacuna": 1.0, "spoke": 1.0, "spot": 1.0}

print('#  엮기(compile)')
model.compile(loss=losses, optimizer='rmsprop', metrics=['accuracy'], loss_weights=lossWeights)

print('[STEP 1] 이미지 불러오기')
datas = []
origs = []

defectLabels = []  # 해당 이미지의 정답(label) 세트를 저장할 list
lacunaLabels = []  # 해당 이미지의 정답(label) 세트를 저장할 list
spokeLabels = []  # 해당 이미지의 정답(label) 세트를 저장할 list
spotLabels = []  # 해당 이미지의 정답(label) 세트를 저장할 list

data_dir = 'D:/Data/iris_pattern/test_image'
imagePaths = sorted(list(paths.list_images(data_dir)))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    print(imagePath)
    image = cv2.imread(imagePath)
    image = findRegion(image)
    image = DataLoader.img_preprocess(image)
    origs.append(image.copy())
    image = img_to_array(image)
    # image = image.astype("float") / 255.0
    # image = np.expand_dims(image, axis=0)
    datas.append(image)

    # b. 이미지 파일명에서  이미지의 정답(label) 세트 추출
    (defect, lacuna, spoke, spot) = imagePath.split(os.path.sep)[-2].split("_")
    defectLabels.append(defect)
    lacunaLabels.append(lacuna)
    spokeLabels.append(spoke)
    spotLabels.append(spot)

data = np.array(datas, dtype="float") / 255.0
defectLabels = np.array(defectLabels)
lacunaLabels = np.array(lacunaLabels)
spokeLabels = np.array(spokeLabels)
spotLabels = np.array(spotLabels)

print("[INFO] binarize both sets of labels..")
from sklearn.preprocessing import LabelBinarizer

# defectLB = LabelBinarizer()
# lacunaLB = LabelBinarizer()
# spokeLB = LabelBinarizer()
# spotLB = LabelBinarizer()
# defectLabels = defectLB.fit_transform(defectLabels)
# lacunaLabels = lacunaLB.fit_transform(lacunaLabels)
# spokeLabels = spokeLB.fit_transform(spokeLabels)
# spotLabels = spotLB.fit_transform(spotLabels)

print('[STEP 3] 모델로 데이터 예측하기')

(defect_predictions, lacuna_predictions, spoke_predictions, spot_predictions) = model.predict(data, batch_size=FLG.BATCH_SIZE)
print(str(len(defect_predictions)))
print('[STEP 4] 결과 보여주기')
for i, (defect_prediction, lacuna_prediction, spoke_prediction, spot_prediction) in enumerate(zip(defect_predictions, lacuna_predictions, spoke_predictions, spot_predictions)):
        preLabels = []
        preLabels.append("{}: {:.2f}%".format('defect', defect_prediction[0] * 100))        # ['defect' 'normal']
        preLabels.append("{}: {:.2f}%".format('lacuna', lacuna_prediction[0] * 100))       # ['lacuna' 'normal']
        preLabels.append("{}: {:.2f}%".format('spoke', spoke_prediction[1] * 100))         # ['normal' 'spoke']
        preLabels.append("{}: {:.2f}%".format('spot', spot_prediction[1] * 100))               # ['normal' 'spot']
        output = imutils.resize(origs[i], width=400)
        y = 300
        for prelabel in preLabels:
            print(prelabel)
            y = y + 20
            cv2.putText(output, prelabel, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, defectLabels[i], (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, lacunaLabels[i], (200, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, spokeLabels[i], (200, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, spotLabels[i], (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Output", output)
        cv2.waitKey(0)


