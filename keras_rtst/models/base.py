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


def add_conv_block(net, name, input_name, filters, filter_size, activation='relu', subsample=(1, 1)):
    net.add_node(Convolution2D(filters, filter_size, filter_size, subsample=subsample, border_mode='same'),
        name + '_conv', input_name)
    net.add_node(BatchNormalization(mode=0, axis=1), name + '_bn',  name + '_conv')
    if isinstance(activation, six.string_types):
        net.add_node(Activation(activation), name,  name + '_bn')
    else:
        net.add_node(activation(), name, name + '_bn')


def create_res_texture_net(input_rows, input_cols, num_res_filters=128,
        res_out_activation='linear', activation='relu', num_res_blocks=5):
    net = Graph()
    net.add_input('x', input_shape=(3, input_rows, input_cols))
    add_conv_block(net, 'in0', 'x', num_res_filters // 4, 9, activation=activation)
    add_conv_block(net, 'in1', 'in0', num_res_filters // 2, 3, subsample=(2, 2), activation=activation)
    add_conv_block(net, 'in2', 'in1', num_res_filters, 3, subsample=(2, 2), activation=activation)
    last_block_name = 'in2'
    for res_i in range(num_res_blocks):
        block_name = 'res_{}'.format(res_i)
        add_conv_block(net, block_name + '_in0', last_block_name, num_res_filters, 3, activation=activation)
        add_conv_block(net, block_name + '_in1', block_name + '_in0', num_res_filters, 3, activation='linear')
        net.add_node(Activation(res_out_activation), block_name, merge_mode='sum', inputs=[block_name + '_in1', last_block_name])
        last_block_name = block_name
    # theano doesn't seem to support fractionally-strided convolutions at the moment
    net.add_node(UpSampling2D(), 'out_up0', last_block_name)
    add_conv_block(net, 'out_0', 'out_up0', num_res_filters // 2, 3, activation=activation)
    net.add_node(UpSampling2D(), 'out_up1', 'out_0')
    add_conv_block(net, 'out_1', 'out_up1', num_res_filters // 4, 3, activation=activation)
    add_conv_block(net, 'out_2', 'out_1', 3, 9, activation='linear')
    net.add_node(Activation('linear'), 'texture_rgb', 'out_2', create_output=True)
    return net


def add_seq_conv_block(net, filters, filter_size, activation='relu', subsample=(1, 1), input_shape=None):
    if input_shape:
        kwargs = dict(input_shape=input_shape)
    else:
        kwargs = dict()
    net.add(Convolution2D(
        filters, filter_size, filter_size, subsample=subsample, border_mode='same', **kwargs))
    net.add(BatchNormalization(mode=0, axis=1))
    if isinstance(activation, six.string_types):
        if activation != 'linear':
            net.add(Activation(activation))
    else:
        net.add(activation())


def create_sequential_texture_net(input_rows, input_cols, num_res_filters=128,
        res_out_activation='linear', activation='relu', num_inner_blocks=5):
    net = Sequential()
    add_seq_conv_block(net, num_res_filters // 4, 9, input_shape=(3, input_rows, input_cols), activation=activation)
    add_seq_conv_block(net, num_res_filters // 2, 3, subsample=(2, 2), activation=activation)
    add_seq_conv_block(net, num_res_filters, 3, subsample=(2, 2), activation=activation)
    for i in range(num_inner_blocks):
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)
    # theano doesn't seem to support fractionally-strided convolutions at the moment
    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 2, 3, activation=activation)
    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 4, 3, activation=activation)
    add_seq_conv_block(net, 3, 9, activation='linear')
    return net


def dumb_objective(x, y):
    '''Returns 0 in a way that makes everyone happy.

    Keras requires outputs and objectives but we're training purely upon the
    loss expressed by the regularizers.
    '''
    return 0.0 * y + 0.0 * x
