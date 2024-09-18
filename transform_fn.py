from imgaug import augmenters as iaa
import numpy as np

class TransformFunction(object):
    def __call__(self, sample):
        seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
            iaa.MultiplySaturation((0.5, 1.2)),
            iaa.MultiplyBrightness((0.5, 1.2)), # no find of exposure
            iaa.GammaContrast((0.5, 1.5)),
            iaa.MotionBlur(k=8), # check if large
        ], random_order=True)

        sample = seq.augment_image(np.array(sample))
        return sample