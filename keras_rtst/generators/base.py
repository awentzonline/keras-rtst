import os
import random

import keras_vgg_buddy
import numpy as np
from keras import backend as K
from scipy.misc import imsave


def output_samples(model, args, input_generator):
    '''Generate args.num_sample outputs drawn from the `input_generator`.'''
    num_samples = args.num_samples
    num_batches = max(num_samples // args.batch_size, 1)
    num_images = 0
    for batch_i in range(num_batches):
        data = next(input_generator)  # this is batch_size'd
        results = model.predict(data, batch_size=args.batch_size)['texture_rgb']
        for i in range(results.shape[0]):
            result = results[i]
            img = keras_vgg_buddy.deprocess_image(result, contrast_percent=args.contrast_percent)
            imsave(args.output_prefix + '_{}.png'.format(num_images), img)
            num_images += 1


def generate_img_batches(root_path, batch_size, resize_shape=None):
    '''Yield random images loaded from `root_path`'''
    img_data = load_images(root_path, resize_shape=resize_shape)
    while True:
        imgs = np.array(random.sample(img_data, batch_size))
        yield imgs


def load_images(root_path, resize_shape=None):
    '''Load up and resize a directory full of images.'''
    data = []
    filenames = sorted(os.listdir(root_path))
    for filename in filenames:
        # TODO: improve vgg-buddy so this isn't necessary
        if resize_shape:
            width = resize_shape[-1]
            square = width == resize_shape[0]
        else:
            width = None
            square = False
        img = keras_vgg_buddy.load_and_preprocess_image(
            os.path.join(root_path, filename),
            width=width, square=square)
        data.append(img)
    return data
