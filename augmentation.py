import random
import numpy as np


class AugmentTransform(object):

    def __init__(self, frequency_shift):
        self.frequency_shift = frequency_shift


    def augment_transform(self, spec_batches):
        augmented_specs = []
        for spec in spec_batches:
            random_freq_shift = random.randint(-self.frequency_shift, self.frequency_shift)
            augmented_spec = np.roll(spec, random_freq_shift, axis=0)
            augmented_specs.extend([augmented_spec])

        return augmented_specs
