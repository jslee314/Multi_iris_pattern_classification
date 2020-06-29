from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, ZeroPadding2D
from keras.layers import Add, LeakyReLU
from keras import backend as K

from CNNModels.Layers.instance_normalization import InstanceNormalization

class LayerBase:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def activation(self, x, type='relu'):
        if type not in ['relu', 'relu6', 'h-swish', 'h-sigmoid', 'sigmoid', 'softmax', 'tanh', 'leaky_relu']:
            raise NotImplementedError(type + ' type is not available. ex) relu, relu6, h-swish, h-sigmoid, sigmoid')

        if type == 'relu':
            return Activation('relu')(x)
        elif type == 'relu6':
            return Activation(self.relu6)(x)
        elif type == 'h-swish':
            return Activation(self.hard_swish)(x)
        elif type == 'h-sigmoid':
            return Activation('hard_sigmoid')(x)
        elif type == 'sigmoid':
            return Activation('sigmoid')(x)
        elif type == 'softmax':
            return Activation('softmax')(x)
        elif type == 'tanh':
            return Activation('tanh')(x)
        elif type == 'leaky_relu':
            return LeakyReLU(alpha=0.01)(x)
        else:
            pass

    def relu6(self, x):
        return K.relu(x, max_value=6)

    def hard_swish(self, x):
        return x * K.relu(x + 3, max_value=6) / 6

    def conv_block(self, x, filters, kernel, strides, scope, nl='relu', norm_type='bn', use_bias=True, pad=0):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        print(scope)
        if pad == 0:
            x = Conv2D(filters, kernel, strides=strides, padding='same', use_bias=use_bias, name=scope)(x)
        else:
            x = ZeroPadding2D(padding=pad)(x)
            x = Conv2D(filters, kernel, strides=strides, padding='valid', use_bias=use_bias, name=scope)(x)

        if norm_type == 'bn' or norm_type == 'batch_norm' or norm_type == 'BN':
            x = BatchNormalization(axis=channel_axis, name=scope + '_bn')(x)
        elif norm_type == 'in' or norm_type == 'instance_norm' or norm_type == 'IN':
            x = InstanceNormalization(axis=channel_axis, name=scope + '_in')(x)
            # pass
        elif norm_type is None or norm_type == '':
            pass
        else:
            raise NotImplementedError(norm_type + ' type is not available. ex) bn(batch_norm), in(instance_norm), None')

        if nl is None:
            return x
        else:
            return self.activation(x, type=nl)

    def deconv_block(self, x, filters, kernel, strides, scope, nl='relu', norm_type='bn', use_bias=False):
        x = UpSampling2D(size=strides, name=scope + '_pool')(x)
        x = self.conv_block(x, filters, kernel, strides=1, nl=nl, norm_type=norm_type, use_bias=use_bias, scope=scope)

        return x

    def se_module(self, x, use_bias=False):
        input_channels = int(x.shape[-1])
        inputs = x

        x = GlobalAveragePooling2D()(x)
        x = Dense(input_channels, activation='relu', use_bias=use_bias)(x)
        x = Dense(input_channels, activation='hard_sigmoid', use_bias=use_bias)(x)
        x = Reshape((1, 1, input_channels))(x)

        return Multiply()([inputs, x])

    def mobilenet_v3_bneck(self, inputs, kernel, exp_size, out, se, nl, scope, strides=1, use_bias=False):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        input_shape = K.int_shape(inputs)
        print(scope)
        residual = strides == 1 and input_shape[3] == out

        x = self.conv_block(inputs, exp_size, 1, strides=1, nl=nl, use_bias=use_bias, scope=scope)

        x = DepthwiseConv2D(kernel,
                            strides=strides,
                            padding='same',
                            depth_multiplier=self.multiplier,
                            use_bias=use_bias,
                            name=scope + '_depthwise')(x)
        x = BatchNormalization(axis=channel_axis, name=scope + '_depthwise_bn')(x)
        x = self.activation(x, type=nl)

        if se:
            x = self.se_module(x, use_bias=use_bias)

        x = Conv2D(int(out * self.multiplier), 1, padding='same', use_bias=use_bias, name=scope + '_pointwise')(x)
        x = BatchNormalization(axis=channel_axis, name=scope + '_pointwise_bn')(x)

        if residual:
            x = Add()([inputs, x])

        return x

    def basic_residual_block(self, inputs, filters, scope, norm_type):
        x = self.conv_block(inputs, filters, 3, strides=1, nl='relu', norm_type=norm_type, pad=1, scope=scope + '_l1')
        x = self.conv_block(x, filters, 3, strides=1, nl=None, norm_type=norm_type, pad=1, scope=scope + '_l2')
        x = Add()([inputs, x])

        return x

    def sepconv_block(self, x, filter, kernel, strides, scope,
                      nl='relu', norm_type='bn', depth_activation=True,
                      use_bias=False, rate=1):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

        if strides == 1:
            padding = 'same'
        else:
            tmp_size = kernel + (kernel - 1) * (rate - 1)
            pad_total = tmp_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            padding = 'valid'

        if not depth_activation:
            self.activation(x, type=nl)
        # Depthwise Conv
        x = DepthwiseConv2D(kernel,
                            strides=strides,
                            padding=padding,
                            dilation_rate=rate,
                            depth_multiplier=self.multiplier,
                            use_bias=use_bias,
                            name=scope + '_depthwise')(x)

        if norm_type == 'bn' or norm_type == 'batch_norm' or norm_type == 'BN':
            x = BatchNormalization(axis=channel_axis, name=scope + '_depthwise_bn')(x)
        elif norm_type == 'in' or norm_type == 'instance_norm' or norm_type == 'IN':
            x = InstanceNormalization(axis=channel_axis, name=scope + '_depthwise_in')(x)
        elif norm_type is None or norm_type == '':
            pass
        else:
            raise NotImplementedError(norm_type + ' type is not available. ex) bn(batch_norm), in(instance_norm)')

        if depth_activation:
            x = self.activation(x, type=nl)

            # Pointwise Conv
            x = self.conv_block(x, filter, 1, 1,
                                scope=scope + '_pointwise',
                                nl=nl,
                                norm_type=norm_type,
                                use_bias=use_bias)
        else:
            # Pointwise Conv
            x = self.conv_block(x, filter, 1, 1,
                                scope=scope + '_pointwise',
                                nl=None,
                                norm_type=norm_type,
                                use_bias=use_bias)

        return x
