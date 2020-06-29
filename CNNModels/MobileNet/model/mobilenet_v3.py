from CNNModels.Layers.layerBase import LayerBase
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape
from keras import backend as K

class MobileNetV3(LayerBase):
    def __init__(self, multiplier, input_shape, num_classes):
        super(MobileNetV3, self).__init__(multiplier)
        self.input_shape = input_shape
        self.num_classes = num_classes

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

