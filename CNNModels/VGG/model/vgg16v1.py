'''
ILSVRC 2014에서 2등,

>>의의
가장 큰 기여는 네트워크 깊이가 좋은성능에 있어서 매유 중요한 요소라는것을 보여주었다.
이들이 제안한 여러개 모델 중 가장 즇은것은 16개의 conv/fc 레이어로 이뤄져있으며, 모든 conv 는 3x3, 모든 pooling은 2x2로 이루어져있다.
 VGG는 동년배 라이벌인 GoogLeNet에 비해 굉장히 간단한 구조를 취하면서도 거의 상이한 퍼포먼스를 보여줬다는 점에서, 신경망의 깊이의 유효성을 증명했다.

 >>단점
 많은 메모리(140M)를 사용하며 많은 연산을 한다는 것이다.
 하지만 네 개의 NVIDIA Titan Black GPU를 사용했음에도 네트워크 하나를 학습시키는데 2~3주가 걸렸다는 비효율의 문제를 안고 있다.

>>특징
(1) 모든 Conv Layer에 3x3 필터 적용
 3x3과 5x5를 사용한 GoogLeNet이나, 7x7 심지어 11x11까지도 사용한 다른 네트워크와는 다르게  고정된 3x3 크기의 필터만을 적용했다.
 7x7 필터와 비교했을 때, 3x3 필터를 세 차례 사용한 결과가 파라미터 수를 81% 줄일 수 있다는 사실에 입각해 큰 필터 사용을 폐기하고 최소 크기의 필터만을 사용했다.
 (3x3이 최소 크기인 이유는 중심 및 상하좌우를 표현할 수 있는 가장 작은 필터이기 때문이다.)

(2) 1x1 Conv Layer 사용
 GoogLeNet과 마찬가지로 'Network in Network'에 영향을 받아 1x1 Conv Layer를 사용했다.
 하지만 이 구조 에서는 연산량 감소를 꾀한다기보다는 의사결정함수에 Non-linearity를 부여할 목적으로 사용되었다.

(3) 다섯 장의 Max-Pooling Layer 사용
 Conv Layer의 수와는 관계없이 다섯 장의 고정된 Pooling Layer만을 사용했으며 적용된 위치는 위 표에서 확인할 수 있다.

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from keras import backend
# Conv2D(컨볼루션 필터수, (컨볼루션 커널 행, 열), activation='relu')

def VGG_16(width, height, depth, classes, weights_path=None):

    input_shape = (height, width, depth)
    chanDim = -1

    if backend.image_data_format() == "channels_first":
        input_shape = (depth, height, width)     #  input_shape=(3, 224, 224)))
        chanDim = 1

    model = Sequential()


    # (CONV  => RELU) * 2 => POOL
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV  => RELU) * 2 => POOL
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV  => RELU) * 3 => POOL
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV  => RELU) * 3 => POOL
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))

    # (CONV  => RELU) * 3 => POOL
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # (FC  => RELU) * 2 => (FC  => Softmax)  * 1
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    return model
