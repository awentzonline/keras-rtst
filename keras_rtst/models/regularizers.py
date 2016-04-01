from keras import backend as K
from keras.regularizers import Regularizer


# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 4
    xs = K.shape(x)
    features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    return gram


class FeatureStyleRegularizer(Regularizer):
    '''Gatys et al 2015 http://arxiv.org/pdf/1508.06576.pdf'''
    def __init__(self, target=None, weight=1.0, **kwargs):
        self.target = target
        self.weight = weight
        super(FeatureStyleRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(gram_matrix(self.target) - gram_matrix(generated)), axis=(1,2))
        ) / (4.0 * K.square(K.prod(K.shape(generated)[1:])))
        return loss


class FeatureContentRegularizer(Regularizer):
    '''Penalizes euclidean distance of content features.'''
    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(FeatureContentRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        content = output[batch_size:, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(content - generated), axis=(1,2,3))
        )
        return loss


class TVRegularizer(Regularizer):
    '''Enforces smoothness in image output.'''
    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(TVRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        x = self.layer.get_output(True)
        assert K.ndim(x) == 4
        a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
        b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
        loss += self.weight * K.mean(K.sum(K.pow(a + b, 1.25), axis=(1,2,3)))
        return loss


class MRFRegularizer(Regularizer):
    '''MRF loss http://arxiv.org/pdf/1601.04589v1.pdf'''
    def __init__(self, features, weight=1.0, patch_size=3, **kwargs):
        self.features = features
        self.weight = weight
        self.patch_size = patch_size
        super(MRFRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        from . import patches

        output = self.layer.get_output(True)
        assert K.ndim(output) == 4
        batch_size = K.shape(output)[0] // 2
        patch_size = self.patch_size
        patch_stride = 1
        generated = output[:batch_size, :, :, :]
        # extract patches from feature maps
        generated_patches, generated_patches_norm = \
            patches.make_patches(generated, patch_size, patch_stride)
        target_patches, target_patches_norm = \
            patches.make_patches(self.features, patch_size, patch_stride)
        # find best patches and calculate loss
        patch_ids = patches.find_patch_matches(
            generated_patches, generated_patches_norm,
            target_patches / target_patches_norm)
        best_target_patches = K.reshape(
            target_patches[patch_ids], K.shape(generated_patches))
        loss += self.weight * K.sum(K.square(best_target_patches - generated_patches)) / patch_size ** 2
        return loss


class AnalogyRegularizer(Regularizer):
    '''Image analogy regularizer'''
    def __init__(self, features_a, features_ap, weight=1.0, patch_size=3, full_analogy=False, **kwargs):
        self.features_a = features_a
        self.features_ap = features_ap
        self.weight = weight
        self.patch_size = patch_size
        self.full_analogy = full_analogy
        super(AnalogyRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        from . import patches

        output = self.layer.get_output(True)
        assert K.ndim(output) == 4
        batch_size = K.shape(output)[0] // 2
        patch_size = self.patch_size
        patch_stride = 1
        generated = output[:batch_size, :, :, :]
        content = output[batch_size:, :, :, :]
        # extract patches from feature maps
        generated_patches, generated_patches_norm = \
            patches.make_patches(generated, patch_size, patch_stride)
        content_patches, content_patches_norm = \
            patches.make_patches(content, patch_size, patch_stride)
        a_patches, a_patches_norm = \
            patches.make_patches(K.variable(self.features_a), patch_size, patch_stride)
        ap_patches, ap_patches_norm = \
            patches.make_patches(K.variable(self.features_ap), patch_size, patch_stride)
        # find best patches and calculate loss
        patch_ids = patches.find_patch_matches(
            content_patches, content_patches_norm,
            a_patches / a_patches_norm)
        best_analogy_patches = K.reshape(
            ap_patches[patch_ids], K.shape(generated_patches))
        loss += self.weight * K.sum(K.square(best_analogy_patches - generated_patches)) / patch_size ** 2
        return loss
