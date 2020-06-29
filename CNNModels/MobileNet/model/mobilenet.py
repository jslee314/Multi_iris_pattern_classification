from keras import backend as K
from keras.backend import backend
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, ZeroPadding2D, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D, Dropout, \
    Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from keras_applications import correct_pad
from CNNModels.Layers.mobilenet_conv import conv_block, depthwise_separable_conv_block, make_divisible, inverted_res_block

BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/releases/download/v0.6/')

models = None
keras_utils = None


class MobileNetBuilder(object):

    @staticmethod
    def build_v1(input_shape,
                  alpha,
                  depth_multiplier,
                  dropout,
                  include_top,
                  input_tensor,
                 pooling,
                 classes):
        # input_tensor에 다른 케라스 모델이나 레이어의 아웃풋이 들어갈 수도 있음

        img_input = Input(shape=input_shape)
        print('img_input  ',  img_input)
        x = conv_block(img_input, 32, alpha, strides=(2, 2))
        print('conv_block  ', x)
        x = depthwise_separable_conv_block(x, 64, alpha, depth_multiplier,
                                            block_id=1)
        print('1  ', x)
        x = depthwise_separable_conv_block(x, 128, alpha, depth_multiplier,  strides=(2, 2),
                                            block_id=2)
        x = depthwise_separable_conv_block(x, 128, alpha, depth_multiplier,
                                            block_id=3)
        x = depthwise_separable_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2),
                                            block_id=4)
        x = depthwise_separable_conv_block(x, 256, alpha, depth_multiplier,
                                            block_id=5)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2),
                                            block_id=6)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier,
                                            block_id=7)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier,
                                            block_id=8)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier,
                                            block_id=9)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier,
                                            block_id=10)
        x = depthwise_separable_conv_block(x, 512, alpha, depth_multiplier,
                                            block_id=11)
        x = depthwise_separable_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2),
                                            block_id=12)
        x = depthwise_separable_conv_block(x, 1024, alpha, depth_multiplier,
                                            block_id=13)

        print(K.image_data_format())

        if include_top:
            if K.image_data_format() == 'channels_first':
                shape = (int(1024 * alpha), 1, 1)
            else:
                shape = (1, 1, int(1024 * alpha))

            x = GlobalAveragePooling2D()(x)
            x = Reshape(shape, name='reshape_1')(x)
            x = Dropout(dropout, name='dropout')(x)
            x = Conv2D(classes, (1, 1),
                              padding='same',
                              name='conv_preds')(x)
            x = Reshape((classes,), name='reshape_2')(x)
            x = Activation('softmax', name='act_softmax')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account  any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = keras_utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # Create model.
        model = Model(inputs, outputs=x)

        return model

    def build_v2(input_shape,
                 alpha,
                 depth_multiplier,
                 dropout,
                 include_top,
                 input_tensor,
                 pooling,
                 classes):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        img_input = Input(shape=input_shape)
        first_block_filters = make_divisible(32 * alpha, 8)

        # If input_shape is None, infer shape from input_tensor
        if input_shape is None and input_tensor is not None:

            try:
                backend.is_keras_tensor(input_tensor)
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is type: ', type(input_tensor),
                                 'which is not a valid type')

            if input_shape is None and not backend.is_keras_tensor(input_tensor):
                default_size = 224
            elif input_shape is None and backend.is_keras_tensor(input_tensor):
                if backend.image_data_format() == 'channels_first':
                    rows = backend.int_shape(input_tensor)[2]
                    cols = backend.int_shape(input_tensor)[3]
                else:
                    rows = backend.int_shape(input_tensor)[1]
                    cols = backend.int_shape(input_tensor)[2]

                if rows == cols and rows in [96, 128, 160, 192, 224]:
                    default_size = rows
                else:
                    default_size = 224

        # If input_shape is None and no input_tensor
        elif input_shape is None:
            default_size = 224

        # If input_shape is not None, assume default size
        else:
            if K.image_data_format() == 'channels_first':
                rows = input_shape[1]
                cols = input_shape[2]
            else:
                rows = input_shape[0]
                cols = input_shape[1]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

        x = ZeroPadding2D(padding=correct_pad(K, img_input, 3), name='Conv1_pad')(img_input)
        x = Conv2D(first_block_filters,  kernel_size=3, strides=(2, 2), padding='valid', use_bias=False, name='Conv1')(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        x = Activation('relu', name='Conv1_relu')(x)

        x = inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

        x = inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
        # x = inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

        x = inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
        # x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
        # x = inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

        x = inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
        # x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
        # x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
        # x = inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)
        #
        # x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
        # x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
        # x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

        x = inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
        # x = inverted_res_block(x, filters=160, alpha=alpha, stride=1,  expansion=6, block_id=14)
        # x = inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

        x = inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_block_filters = make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280

        x = Conv2D(last_block_filters,
                          kernel_size=1,
                          use_bias=False,
                          name='Conv_1')(x)
        x = BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name='Conv_1_bn')(x)
        x = Activation('relu', name='out_relu')(x)

        if include_top:
            x = GlobalAveragePooling2D()(x)
            x = Dense(classes, activation='softmax',
                             use_bias=True, name='Logits')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        if input_tensor is not None:
            inputs = keras_utils.get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, outputs=x)

        return model


    def build(self, config):
        inputs = Input(shape=self.input_shape)

        if config not in ['large', 'small']:
            raise NotImplementedError(config + ' is not available.')

        channel_before_pool = 960 if config == 'large' else 576
        channel_after_pool = 1280 if config == 'large' else 1024

        cfg = self.get_config(config=config)

        x = self.conv_block(inputs, 16, 3, strides=2, nl='relu6', scope='conv1')

        # MobileNetV3 Bottleneck Block.
        naming = ['bneck{}'.format(name + 1) for name in range(len(cfg))]

        for i, ((k, exp, out, se, nl, stride), name) in enumerate(zip(cfg, naming)):
            x = self.mobilenet_v3_bneck(x, k, exp, out, se, nl, stride, scope=name)

        x = self.conv_block(x, channel_before_pool, 1, strides=1, nl='h-swish', scope='conv2')
        x = GlobalAveragePooling2D()(x)

        c = K.int_shape(x)[-1]

        x = Reshape((1, 1, c))(x)
        x = self.conv_block(x, channel_after_pool, 1, strides=1, nl='h-swish', bn=None, scope='conv3')
        x = self.conv_block(x, self.num_classes, 1, strides=1, nl='softmax', bn=None, scope='conv4')

        x = Reshape((self.num_classes,))(x)

        model = Model(inputs, x)

        return model

    def get_config(self, config='large'):
        if config == 'large':
            cfg = [
                # k, exp, out, se, nl, stride
                [3, 16, 16, False, 'relu6', 1],
                [3, 64, 24, False, 'relu6', 2],
                [3, 72, 24, False, 'relu6', 1],
                [5, 72, 40, True, 'relu6', 2],
                [5, 120, 40, True, 'relu6', 1],
                [5, 120, 40, True, 'relu6', 1],
                [3, 240, 80, False, 'h-swish', 2],
                [3, 200, 80, False, 'h-swish', 1],
                [3, 184, 80, False, 'h-swish', 1],
                [3, 184, 80, False, 'h-swish', 1],
                [3, 480, 112, True, 'h-swish', 1],
                [3, 672, 112, True, 'h-swish', 1],
                [5, 672, 160, True, 'h-swish', 2],
                [5, 960, 160, True, 'h-swish', 1],
                [5, 960, 160, True, 'h-swish', 1]
            ]
        elif config == 'small':
            cfg = [
                # k, exp, out, se, nl, stride
                [3, 16, 16, True, 'relu6', 2],
                [3, 72, 24, False, 'relu6', 2],
                [3, 88, 24, False, 'relu6', 1],
                [5, 96, 40, True, 'h-swish', 2],
                [5, 240, 40, True, 'h-swish', 1],
                [5, 240, 40, True, 'h-swish', 1],
                [5, 120, 48, True, 'h-swish', 1],
                [5, 144, 48, True, 'h-swish', 1],
                [5, 288, 96, True, 'h-swish', 2],
                [5, 576, 96, True, 'h-swish', 1],
                [5, 576, 96, True, 'h-swish', 1]
            ]
        else:
            raise ImportError

        return cfg

    @staticmethod
    def build_mobilenet_v1(input_shape, classes):

        return MobileNetBuilder.build_v1(
            input_shape=input_shape,
            alpha=1.0,
            depth_multiplier=1,
            dropout=1e-3,
            include_top=True,
            input_tensor=None,
            pooling='max',
            classes=classes)

    @staticmethod
    def build_mobilenet_v2(input_shape, classes):
        return MobileNetBuilder.build_v2(
            input_shape=input_shape,
            alpha=1.0,
            depth_multiplier=1,
            dropout=1e-3,
            include_top=True,
            input_tensor=None,
            pooling=None,
            classes=classes)

    @staticmethod
    def build_mobilenet_v3(input_shape, classes):
        return MobileNetBuilder.build_v3(
            input_shape=input_shape,
            alpha=1.0,
            depth_multiplier=1,
            dropout=1e-3,
            include_top=True,
            weights='imagenet',
            input_tensor=None, pooling=None,
            classes=classes)