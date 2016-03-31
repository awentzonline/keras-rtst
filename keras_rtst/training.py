import glob
import os
import random
import time

import numpy as np
import keras_vgg_buddy
from keras import backend as K
from keras.callbacks import LearningRateScheduler


def generate_training_batches(args, input_generator):
    '''Prepare a batch of input data from input_generator to be used as training data.
    Essentially it's just satisfying the `dumb_objective` for `texture_rgb`
    '''
    dumb_output = dict(texture_rgb=np.zeros((args.batch_size, 3, args.max_width, args.max_width)))
    while True:
        data = next(input_generator)
        data.update(dumb_output)
        yield data


class TextureNetLearningSchedule(object):
    def __init__(self, batch_size, num_iterations_per_epoch, initial_lr=0.1, min_lr=0.001, cliff=10, falloff=0.9):
        self.batch_size = batch_size
        self.num_iterations_per_epoch = num_iterations_per_epoch
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.cliff = cliff
        self.falloff = falloff

    def __call__(self, epoch):
        cliff_epoch = self.cliff # // self.num_iterations_per_epoch
        if epoch < cliff_epoch:  # run hot for a while
            new_lr = self.initial_lr
        else:
            new_lr = self.initial_lr * self.falloff ** ((epoch - cliff_epoch) + 1)
        new_lr = max(new_lr, self.min_lr)
        print('New learning rate: {}'.format(new_lr))
        return new_lr


def train(model, args, input_generator, callbacks=[], num_samples=10):
    lr_schedule = TextureNetLearningSchedule(
        args.batch_size, args.num_iterations_per_epoch,
        initial_lr=args.learn_rate, min_lr=args.min_learn_rate,
        cliff=20)
    callbacks.append(LearningRateScheduler(lr_schedule))
    try:
        model.fit_generator(
            generate_training_batches(args, input_generator),
            samples_per_epoch=args.num_iterations_per_epoch, nb_epoch=args.num_epochs,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print('Stopping training...')
