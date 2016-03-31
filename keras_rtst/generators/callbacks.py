from keras.callbacks import Callback

from .base import output_samples


class GenerateSamplesCallback(Callback):
    '''Generate some samples every epoch.'''
    def __init__(self, model, args, input_generator, verbose=0):
        super(GenerateSamplesCallback, self).__init__()
        self.model = model
        self.args = args
        self.input_generator = input_generator
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        output_samples(self.model, self.args, self.input_generator)
