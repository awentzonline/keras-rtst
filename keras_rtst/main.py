import os
import time

import numpy as np
import keras_vgg_buddy

from .generators.base import output_samples
from .generators.style_xfer import output_size_from_glob, transform_glob
from .generators.callbacks import GenerateSamplesCallback
from .models.config import get_model_by_name
from .training import train


def main(args):
    # ensure output path exists
    output_dir = os.path.dirname(args.output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.train:
        train_main(args)
    else:
        generate_main(args)


def train_main(args):
    # should this allow training on rectangles or just squares?
    args.max_height = args.max_width
    assert args.style_image_path, 'Style image path required for training.'
    style_img = keras_vgg_buddy.load_and_preprocess_image(args.style_image_path, width=args.max_width, square=True)

    print('creating model...')
    model, input_generator, eval_generator = get_model_by_name(args.model)
    model = model(args, style_img=style_img)
    print('loading weights...')
    weights_filename = args.weights_prefix + '.weights'
    model_filename = args.weights_prefix + '.json'
    if not args.ignore_weights and os.path.exists(weights_filename):
        model.nodes['texnet'].load_weights(weights_filename)

    input_generator = input_generator(args)
    eval_generator = eval_generator(args)
    started_training = time.time()
    train(model, args, input_generator,
        callbacks=[GenerateSamplesCallback(model, args, eval_generator)]
    )
    print('Done training after {:.2f} seconds'.format(time.time() - started_training))
    print('saving weights')
    save_kwargs = {}
    if args.auto_save_weights:
        save_kwargs['overwrite'] = True
    model.nodes['texnet'].save_weights(weights_filename, **save_kwargs)
    model_json = model.nodes['texnet'].to_json()
    if args.save_model:
        with open(model_filename, 'w') as model_file:
            model_file.write(model_json)
    # output final samples
    output_samples(model, args, eval_generator)


def generate_main(args):
    # determine input image size
    args.max_height, args.max_width = output_size_from_glob(args.convert_glob, width=args.max_width)
    print('creating model...')
    model, input_generator, eval_generator = get_model_by_name(args.model)
    model = model(args)
    print('loading weights...')
    weights_filename = args.weights_prefix + '.weights'
    if not args.ignore_weights and os.path.exists(weights_filename):
        model.nodes['texnet'].load_weights(weights_filename)

    transform_glob(model, args)


if __name__ == '__main__':
    import argparser
    import os

    args = argparser.get_parser()

    output_dir = os.path.dirname(args.output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(args)
