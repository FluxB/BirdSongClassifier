import random
import numpy as np
import math
import matplotlib.pyplot as plt


class AugmentTransform(object):

    def __init__(self, frequency_shift, time_shift):
        self.frequency_shift = frequency_shift
        self.time_shift = time_shift
        self.inverse_labels = {}


    def configure_same_class_augmentation(self, inverse_labels, preprocessor, samples_to_add=[1, 2]):
        self.inverse_labels = inverse_labels
        self.preprocessor = preprocessor
        self.samples_to_add = samples_to_add


    def augment_transform(self, spec_batches, labels):
        augmented_specs = []
        for spec, label in zip(spec_batches, labels):
            augmented_spec = self.freq_augmentation(spec)
            augmented_spec = self.time_augmentation(augmented_spec)
            if label in self.inverse_labels.keys():
                augmented_spec = self.same_class_augmentation(augmented_spec, label)
            augmented_specs.extend([augmented_spec])

        return augmented_specs


    def freq_augmentation(self, spec):
        random_freq_shift = random.randint(-self.frequency_shift, self.frequency_shift)
        augmented_spec = np.roll(spec, random_freq_shift, axis=0)
        return augmented_spec


    def time_augmentation(self, spec):
        random_time_shift = random.randint(-self.time_shift, self.time_shift)
        augmented_spec = np.roll(spec, random_time_shift, axis=1)
        return augmented_spec


    def same_class_augmentation(self, spec, label):
        augmented_spec = spec
        same_class_pool = self.inverse_labels[label]
        (lower_to_add, upper_to_add) = self.samples_to_add
        nr_to_add = random.randint(lower_to_add, upper_to_add)
        for i in range(nr_to_add):
            r = random.randint(0, len(same_class_pool) - 1)
            path = same_class_pool[r]
            spec_to_add = self.preprocessor.load_and_preprocess([path])[0]
            augmented_spec = self.__add_two_specs(augmented_spec, spec_to_add)
            augmented_spec /= np.max(augmented_spec)
        
        return augmented_spec


    def __add_two_specs(self, spec1, spec2):
        (f1, t1) = spec1.shape
        (f2, t2) = spec2.shape
        if t1 < t2:
            nr_repeat = math.ceil(t2 / t1)
            spec1 = np.repeat(spec1, nr_repeat, axis=1)
            spec1 = spec1[:, :t2]
        elif t2 < t1:
            nr_repeat = math.ceil(t1 / t2)
            spec2 = np.repeat(spec2, nr_repeat, axis=1)
            spec2 = spec2[:, :t1]

        return (spec1 + spec2)
        
