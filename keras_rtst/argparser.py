import argparse
import os


VGG_ENV_VAR = 'VGG_WEIGHT_PATH'


class CommaSplitAction(argparse.Action):
    '''Split n strip incoming string argument.'''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [v.strip() for v in values.strip().split(',') if v.strip()])


def get_args():
    parser = argparse.ArgumentParser(description='Real-Time Style Transfer with Keras')
    parser.add_argument('output_prefix', metavar='output_prefix', type=str,
                        help='Prefix for output')
    parser.add_argument('--style-img', dest='style_image_path', type=str,
                        help='Path to the style image.')
    parser.add_argument('--style-map-img', dest='style_map_image_path', type=str,
                        default=None, help='Path to the style map image for analogy loss (A)')
    parser.add_argument('--max-width', dest='max_width', type=int,
                        default=256, help='Max width')
    parser.add_argument('--max-height', dest='max_height', type=int,
                        default=None, help='Max height. Just leave it alone for now.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        default=4, help='Batch size')
    parser.add_argument('--num-samples', dest='num_samples', type=int,
                        default=16, help='How many samples to generate every training epoch.')
    parser.add_argument('--weights-prefix', dest='weights_prefix', type=str,
                        default='texnet', help='Prefix for texturenet weight filenames')
    parser.add_argument('--ignore-weights', dest='ignore_weights', action='store_true',
                        help='Do not load any existing weights.')
    parser.add_argument('--auto-save-weights', dest='auto_save_weights', action='store_true',
                        help='Save over existing weights without asking.')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train network')
    parser.add_argument('--convert-glob', dest='convert_glob', type=str,
                        default='', help='Glob for files to convert')
    parser.add_argument('--train-data', dest='training_data_path', type=str,
                        default='./training-images/', help='Path to training images.')
    parser.add_argument('--eval-data', dest='eval_data_path', type=str,
                        default='./training-images/', help='Path to evaluation images.')
    parser.add_argument('--contrast', dest='contrast_percent', type=float,
                        default=0.0, help='Stretch contrast percentile.')
    parser.add_argument('--sequential-model', dest='sequential_model', action='store_true',
                        help='Use a sequential rather than residual/graph model.')
    parser.add_argument('--activation', type=str, default='LeakyReLU',
                        help='Activation function to use.')
    parser.add_argument('--num-res-filters', type=int, default=128,
                        help='Number of filters on the convolutional layers in the residual blocks')
    parser.add_argument('--num-blocks', type=int, default=5,
                        help='Number of repeated inner residual blocks (default=5)')
    parser.add_argument('--depth', type=int, default=3,
                        help='Max depth for girthy model (default=3)')
    # losses
    parser.add_argument('--content-w', dest='content_weight', type=float,
                        default=0.01, help='Content loss weight')
    parser.add_argument('--content-layers', dest='content_layers', action=CommaSplitAction,
                        default=['conv2_2'],
                        help='Comma-separated list of layer names to be used for the content loss')
    parser.add_argument('--style-w', dest='style_weight', type=float,
                        default=10.0, help='Style loss weight')
    parser.add_argument('--style-layers', dest='style_layers', action=CommaSplitAction,
                        default=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'],
                        help='Comma-separated list of layer names to be used for the content loss')
    parser.add_argument('--mrf-w', dest='mrf_weight', type=float,
                        default=0.0, help='MRF loss weight')
    parser.add_argument('--mrf-layers', dest='mrf_layers', action=CommaSplitAction,
                        default=['conv4_2'],
                        help='Comma-separated list of layer names to be used for the MRF loss')
    parser.add_argument('--analogy-w', dest='analogy_weight', type=float,
                        default=0.0, help='Image analogy loss weight')
    parser.add_argument('--analogy-layers', dest='analogy_layers', action=CommaSplitAction,
                        default=['conv4_1'],
                        help='Comma-separated list of layer names to be used for the analogy loss')
    parser.add_argument('--tv-w', dest='tv_weight', type=float,
                        default=0.00001, help='Texture total variation loss (smoothness).')
    # vgg
    parser.add_argument('--vgg-weights', dest='vgg_weights', type=str,
                        default=os.environ.get(VGG_ENV_VAR, 'vgg16_weights.h5'), help='Path to VGG16 weights.')
    parser.add_argument('--pool-mode', dest='pool_mode', type=str,
                        default='max', help='Pooling mode for VGG ("avg" or "max")')
    # optimizer
    parser.add_argument('--learn-rate', dest='learn_rate', type=float,
                        default=0.1, help='Initial learning rate')
    parser.add_argument('--min-learn-rate', dest='min_learn_rate', type=float,
                        default=0.001, help='Final learning rate')
    parser.add_argument('--iters', dest='num_iterations_per_epoch', type=int,
                        default=100, help='Number of iterations per epoch')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=400, help='Number of training epochs')
    parser.add_argument('--model', dest='model', type=str,
                        default='transfer', help='Which model to use. Only supports "transfer"')
    parser.add_argument('--save-model', dest='save_model', action='store_true',
                        help='Serialize the model as JSON')
    return parser.parse_args()
