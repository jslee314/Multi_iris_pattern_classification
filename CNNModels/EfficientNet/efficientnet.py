import os, math

from tensorflow.keras import backend as K
from keras.utils import get_file
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras_applications.imagenet_utils import preprocess_input as _preprocess
from keras_applications.imagenet_utils import _obtain_input_shape

from CNNModels.EfficientNet.layerBase import LayerBase

'''
모든 소스는 아래와 같은 github에서 참조하여 작성되었습니다.
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
'''

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def efficientNet_factory(model_name, load_weights=None, input_shape=None, classes=1000):

    factory = {
        'efficientnet-b0': EfficientNet(224, 1.0, 1.0, 0.2, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b0'),

        'efficientnet-b1': EfficientNet(240, 1.0, 1.1, 0.2, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b1'),

        'efficientnet-b2': EfficientNet(260, 1.1, 1.2, 0.3, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b2'),

        'efficientnet-b3': EfficientNet(300, 1.2, 1.4, 0.3, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b3'),

        'efficientnet-b4': EfficientNet(380, 1.4, 1.8, 0.4, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b4'),

        'efficientnet-b5': EfficientNet(456, 1.6, 2.2, 0.4, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b5'),

        'efficientnet-b6': EfficientNet(528, 1.8, 2.6, 0.5, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b6'),

        'efficientnet-b7': EfficientNet(600, 2.0, 3.1, 0.5, load_weights, classes,
                                        input_shape=input_shape, name='efficientnet-b7'),
    }

    default_size = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 380,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
    }

    return factory[model_name].efficientNet(), default_size[model_name]

class EfficientNet(LayerBase):
    def __init__(self,
                 default_size,
                 width_coefficient: float,
                 depth_coefficient: float,
                 dropout_rate=0.,
                 load_weights=None,
                 classes=1000,
                 input_tensor=None,
                 input_shape=None,
                 drop_connect=0.2,
                 depth_divisor=8,
                 min_depth=None,
                 multiplier=1,
                 name='EfficientNet'):
        self.default_size = default_size
        self.input_shape = input_shape
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.load_weights = load_weights
        self.input_tensor = input_tensor
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.drop_connect = drop_connect
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.multiplier = multiplier
        self.name = name
        super(EfficientNet, self).__init__(multiplier)

    def preprocess_input(self, x, data_format='channel_last'):
        return _preprocess(x, data_format, mode='torch')

    def round_filter(self, filters, width_coefficient, depth_divisor, min_depth):
        multiplier = float(width_coefficient)
        divisor = int(depth_divisor)
        min_depth = min_depth

        if not multiplier:
            return filters

        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor

        return int(new_filters)

    def round_repeat(self, repeats, depth_coefficient):
        multiplier = depth_coefficient
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def efficientNet(self):
        input_shape = _obtain_input_shape(self.input_shape,
                                          default_size=self.default_size,
                                          data_format=K.image_data_format(),
                                          min_size=32,
                                          require_flatten=True,
                                          weights=self.load_weights)

        if self.input_tensor is None:
            image_input = Input(shape=input_shape)
        else:
            image_input = self.input_tensor

        cfg = self.get_config()
        num_block = sum(c[0] for c in cfg)
        drop_connect = self.drop_connect / float(num_block)
        w_coef, d_coef, m_depth = self.width_coefficient, self.depth_coefficient, self.min_depth

        # Stem Layer
        x = self.conv_block(image_input,
                            filters=self.round_filter(32, w_coef, d_coef, m_depth),
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            kernel=3, strides=2, scope='stem_conv1',
                            nl='swish', use_bias=False)

        # MBConv Block
        for idx, (repeat, kernel, exp, out, se, nl, stride, expand_ratio) in enumerate(cfg):
            exp = self.round_filter(exp, w_coef, d_coef, m_depth)
            out = self.round_filter(out, w_coef, d_coef, m_depth)
            if not (idx == 0 or idx == len(cfg) - 1):
                repeat = self.round_repeat(repeat, self.depth_coefficient)

            x = self.mobilenet_v3_bneck(x, kernel, exp, out, se, nl,
                                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                                        dense_initializer=DENSE_KERNEL_INITIALIZER,
                                        scope='block%d_mbconv1' % (idx + 1),
                                        strides=stride,
                                        expand_ratio=expand_ratio,
                                        drop_connect=drop_connect * idx)

            if repeat > 1:
                exp = out
                stride = 1

            for i in range(repeat - 1):
                x = self.mobilenet_v3_bneck(x, kernel, exp, out, se, nl,
                                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                                            dense_initializer=DENSE_KERNEL_INITIALIZER,
                                            scope='block%d_mbconv%d' % ((idx + 1), (i + 2)),
                                            strides=stride,
                                            expand_ratio=expand_ratio,
                                            drop_connect=drop_connect * idx)

        # Head Layer
        x = self.conv_block(x,
                            filters=self.round_filter(1280, w_coef, d_coef, m_depth),
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            kernel=1, strides=1, scope='head_conv1',
                            nl='swish', use_bias=False)
        x = GlobalAveragePooling2D()(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Dense(self.classes, kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
        x = self.activation(x, type='softmax')

        outputs = x
        model = Model(image_input, outputs, name=self.name)

        # Load Weight
        if self.load_weights is not None:
            model.load_weights(self.load_weights)

        return model

    def get_config(self):
        dimRatio = 8    # default 는 8
        in_dims = [4, 2, 3, 5, 10, 14, 24]

        out_dims = [2, 3, 5, 10, 14, 24, 40]
        cfg = [
            # [repeat, k, exp, out, se, nl, stride, expand_ratio]
            # repeat : 레이어 반복 횟수
            #  k : 커널 크기
            # exp  : 블럭  input channel
            # out : 블럭   output chanel
            # se : squeeze and excite module
            # nl :  활성함수
            # stride: 스트라이드
            # expand_ratio :  channel 의 팽창계수(mobileNetV3)
            [1, 3, in_dims[0]*dimRatio, out_dims[0]*dimRatio, True, 'swish', 1, 1],
            [2, 3, in_dims[1]*dimRatio, out_dims[1]*dimRatio, True, 'swish', 2, 6],
            [2, 5, in_dims[2]*dimRatio, out_dims[2]*dimRatio, True, 'swish', 2, 6],
            [3, 3, in_dims[3]*dimRatio, out_dims[3]*dimRatio, True, 'swish', 2, 6],
            [3, 5, in_dims[4]*dimRatio, out_dims[4]*dimRatio, True, 'swish', 1, 6],
            [4, 5, in_dims[5]*dimRatio, out_dims[5]*dimRatio, True, 'swish', 2, 6],
            [1, 3, in_dims[6]*dimRatio, out_dims[6]*dimRatio, True, 'swish', 1, 6]
        ]
        return cfg