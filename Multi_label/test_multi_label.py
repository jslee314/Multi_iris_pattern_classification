import keras
from keras.models import load_model
import pickle
import cv2
from Multi_label.util.constants import *


# # keras session 설정
# config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5))
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)

# 1) Data set 생성
image = cv2.imread('examples/example_01.jpg')
output = image.copy()
image = cv2.resize(image, (FLG.WIDTH, FLG.HEIGHT))
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # reshape((1, image.shape[0])) 대신


# 2)  load the modelsaved & label binarizer
print('h5 :   ' +FLG.H5 )
model = load_model(FLG.H5)								# output/simple_nn.modelsaved
lb = pickle.loads(open(FLG.LANEL_BIN, "rb").read())             # output/simple_nn_lb.pickle


#  3)  predict
preds = model.predict(image) # prdes = (0.05, 0.89, 0.07)


#  4)  voting
# 가장 큰 확률을 가진 클래스 레이블을 찾음
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

#  draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 125, 127), 2)
cv2.imshow("Image", output)
cv2.waitKey(0)
