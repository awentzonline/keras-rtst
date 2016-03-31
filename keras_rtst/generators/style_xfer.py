import glob
import os

import keras_vgg_buddy
import numpy as np
from keras import backend as K
from scipy.misc import imsave

from .base import generate_img_batches


def input_generator(args):
    '''Generates batches of random samples paired with images.
    '''
    g_training_imgs = generate_img_batches(
        args.training_data_path, args.batch_size,
        resize_shape=(args.max_height, args.max_width))
    while True:
        data = {
            'content': np.array(next(g_training_imgs))
        }
        yield data


def evaluation_input_generator(args):
    '''Generates batches of random samples paired with images.
    '''
    g_training_imgs = generate_img_batches(
        args.eval_data_path, args.batch_size,
        resize_shape=(args.max_height, args.max_width))
    while True:
        data = {
            'content': np.array(next(g_training_imgs))
        }
        yield data


def transform_glob(model, args):
    '''Apply the model to a glob of images.'''
    f_generate = K.function([model.inputs['content'].input],
        [model.nodes['texnet'].get_output(False)])
    filenames = glob.glob(args.convert_glob)
    output_path = args.output_prefix
    try:
        os.makedirs(output_path)
    except OSError:
        pass  # exists
    for filename in filenames:
        print('converting {}'.format(filename))
        img = keras_vgg_buddy.load_and_preprocess_image(filename, width=args.max_width)
        result = f_generate([np.expand_dims(img, 0)])[0]
        img = keras_vgg_buddy.deprocess_image(result[0], contrast_percent=0)
        imsave(os.path.join(output_path, os.path.basename(filename)), img)


def output_size_from_glob(target_glob, width=256):
    filenames = glob.glob(target_glob)
    assert filenames, 'No files matched glob: {}'.format(target_glob)
    img0 = keras_vgg_buddy.load_and_preprocess_image(filenames[0], width=width)
    return img0.shape[-2:]
