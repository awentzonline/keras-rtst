import os
import time

import numpy as np
import six
import keras.initializations
import keras_vgg_buddy
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Layer
from keras.layers.convolutional import AveragePooling2D, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Graph, Sequential


def add_conv_block(net, name, input_name, filters, filter_size, activation='relu',
        subsample=(1, 1), init='glorot_uniform'):
    net.add_node(
        Convolution2D(filters, filter_size, filter_size,
            subsample=subsample, border_mode='same', init=init),
        name + '_conv', input_name)
    net.add_node(BatchNormalization(mode=0, axis=1), name + '_bn',  name + '_conv')
    if isinstance(activation, six.string_types):
        net.add_node(Activation(activation), name,  name + '_bn')
    else:
        net.add_node(activation(), name, name + '_bn')


def add_conv_block_group(g, name, input_name, num_filters, filter_size):
    add_model_block(g, name + '_b0', input_name, num_filters, filter_size)
    add_model_block(g, name + '_b1', name + '_b0', num_filters, filter_size)
    add_model_block(g, name, name + '_b1', num_filters, 1)


def create_res_texture_net(input_rows, input_cols, num_res_filters=128,
        res_out_activation='linear', activation='relu', num_res_blocks=5, depth=3):
    '''Adds a series of residual blocks at each resolution scale, rather than just
    the minimium one.
    '''
    net = Graph()
    net.add_input('x', input_shape=(3, input_rows, input_cols))
    add_conv_block(net, 'in0', 'x', num_res_filters // 4, 9, activation=activation)
    last_name = 'in0'
    # scale down input to max depth with a series of strided convolutions
    for scale_i in range(depth):
        num_scale_filters = num_res_filters - scale_i * 8 # // (2 ** scale_i) # (depth - scale_i - 1))
        scale_name = 'down_{}'.format(scale_i)
        add_conv_block(net, scale_name, last_name, num_scale_filters, 3, subsample=(2, 2), activation=activation)
        last_name = scale_name
    # add a series of residual blocks at each scale, from smallest to largest
    for scale_i in reversed(range(depth)):
        num_scale_filters = num_res_filters - scale_i * 8 # // (2 ** scale_i) # (depth - scale_i - 1))
        last_scale_name = last_name
        for res_i in range(num_res_blocks):
            block_name = 'res_{}_{}'.format(scale_i, res_i)
            add_conv_block(net, block_name + '_b0', last_name, num_res_filters, 3, activation=activation)
            add_conv_block(net, block_name + '_b1', block_name + '_b0', num_res_filters, 1, activation='linear')
            if last_name == last_scale_name:
                # tranform residual connection to same number of filters
                add_conv_block(net, block_name + '_res', last_name, num_res_filters, 1, activation='linear')
            else:
                # no transform needed when the last node was part of the current residual block
                net.add_node(Layer(), block_name + '_res', last_name)
            net.add_node(Activation(res_out_activation), block_name, merge_mode='sum', inputs=[block_name + '_b1', block_name + '_res'])
            last_name = block_name
        # theano doesn't seem to support fractionally-strided convolutions at the moment
        up_name = 'up_{}'.format(scale_i)
        net.add_node(UpSampling2D(), up_name, last_name)
        last_name = up_name
        last_scale_name = up_name
    # final output
    add_conv_block(net, 'out', last_name, 3, 9, activation='linear')
    net.add_node(Activation('linear'), 'texture_rgb', 'out', create_output=True)
    return net
