
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

def mybuild(width, height, depth, classes):
    model = Sequential()

    inputShape = (height, width, depth)
    # chanDim = -1
    # # if we are using "channels first", update the input shape and channels dimension
    # if K.image_data_format() == "channels_first":
    #     inputShape = (depth, height, width)
    #     chanDim = 1

    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    ''' binary classification or muliobject classification '''
    if classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary__crossentropy', optimizer='sgd', metrics=['accuracy'])
    else:
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model
