'''Texture Network for style transfer.'''
import time

import keras_vgg_buddy
import numpy as np
from keras import backend as K
from keras.layers.core import Layer
from keras.layers.convolutional import AveragePooling2D
from keras.models import Graph
from keras.optimizers import Adam

from .base import create_res_texture_net, dumb_objective
from .regularizers import (
    FeatureContentRegularizer, FeatureStyleRegularizer, MRFRegularizer, TVRegularizer)


def make_model(args, style_img=None, activation='relu', pool_class=AveragePooling2D):
    model = Graph()
    model.add_input('content', batch_input_shape=(args.batch_size, 3, args.max_height, args.max_width))
    texnet = create_res_texture_net(args.max_height, args.max_width)
    # add the texture net to the model
    model.add_node(texnet, 'texnet', 'content')
    model.add_output('texture_rgb', 'texnet')
    # hook up the training network stuff
    if args.train:
        model.add_node(Layer(), 'vgg_concat', inputs=['texnet', 'content'], concat_axis=0)
        # add VGG and the constraints
        keras_vgg_buddy.add_vgg_to_graph(model, 'vgg_concat', pool_mode=args.pool_mode,
            trainable=False, weights_path=args.vgg_weights)
        # add the regularizers for the various feature layers
        vgg = keras_vgg_buddy.VGG16(args.max_height, args.max_width, pool_mode=args.pool_mode, weights_path=args.vgg_weights)
        print('computing static style features')
        style_features = vgg.get_features(np.expand_dims(style_img, 0), set(args.style_layers + args.content_layers + args.mrf_layers))
        regularizers = []
        if args.style_weight != 0.0:
            for layer_name in args.style_layers:
                layer = model.nodes[layer_name]
                style_regularizer = FeatureStyleRegularizer(
                    target=style_features[layer_name], weight=args.style_weight,
                    num_inputs=2)
                style_regularizer.set_layer(layer)
                regularizers.append(style_regularizer)
        if args.content_weight != 0.0:
            for layer_name in args.content_layers:
                layer = model.nodes[layer_name]
                content_regularizer = FeatureContentRegularizer(weight=args.content_weight)
                content_regularizer.set_layer(layer)
                regularizers.append(content_regularizer)
        if args.mrf_weight != 0.0:
            for layer_name in args.mrf_layers:
                layer = model.nodes[layer_name]
                mrf_regularizer = MRFRegularizer(K.variable(style_features[layer_name]), weight=args.mrf_weight)
                mrf_regularizer.set_layer(layer)
                regularizers.append(mrf_regularizer)
        if args.tv_weight != 0.0:
            tv_regularizer = TVRegularizer(weight=args.tv_weight)
            tv_regularizer.set_layer(model.nodes['texnet'])
            regularizers.append(tv_regularizer)
        setattr(model.nodes['vgg_concat'], 'regularizers', regularizers)  # Gotta put em somewhere?

    print('compiling')
    start_compile = time.time()
    adam = Adam(lr=args.learn_rate, beta_1=0.7)
    model.compile(optimizer=adam, loss=dict(texture_rgb=dumb_objective))
    print('Compiled model in {:.2f}'.format(time.time() - start_compile))
    return model
