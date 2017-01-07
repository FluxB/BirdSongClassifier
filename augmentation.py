import random
import numpy as np
import math
import matplotlib.pyplot as plt


class AugmentTransform(object):

    # frequency_shift: maximal number of pixels for frequency augmentation
    # time_shift: maximal number of pixels for time augmentation
    def __init__(self, frequency_shift, time_shift):
        self.frequency_shift = frequency_shift
        self.time_shift = time_shift
        self.inverse_labels = {}
        self.inverse_labels_bg = {}
        self.max_bg_intensity = 40 #in %
    
    # this is necessary to call before same_class_augmentation()
    # it takes the inverse lookup structure such that we can find 
    # same class samples. 
    # It takes a preprocessor to preprocess the sample we augment with
    # samples_to_add: lower and upper bound for the number of samples with
    # which we augment.
    def configure_same_class_augmentation(self, inverse_labels, label_bg_path, preprocessor, samples_to_add=[1, 2]):
        # self.inverse_labels_bg = inverse_labels_bg
        self.label_bg_path = label_bg_path
        self.inverse_labels = inverse_labels
        self.preprocessor = preprocessor
        self.samples_to_add = samples_to_add


    # main function. augments any spec from spec_batches with all available augment
    # functions. labels holds the respective labels of each spec, which is necessary for
    # for same_class_augmentation
    def augment_transform(self, spec_batches, labels):
        augmented_specs = []
        for spec, label in zip(spec_batches, labels):
            augmented_spec = spec
            augmented_spec = self.freq_augmentation(spec)
            augmented_spec = self.time_augmentation(augmented_spec)
            if label in self.inverse_labels.keys():
                augmented_spec = self.same_class_augmentation(augmented_spec, label)
            if label in self.inverse_labels_bg.keys():
                augmented_spec = self.bg_augmentation(augmented_spec, label)
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

    def bg_augmentation(self, spec, label):
        # same_class_bg_pool = self.inverse_labels_bg[label]
        r = random.randint(0, len(self.label_bg_path) - 1)
        intensity = float(random.randint(0, self.max_bg_intensity))/100
        path = self.label_bg_path[r]  # same_class_bg_pool[r]
        bg_to_add = intensity * self.preprocessor.load_and_preprocess([path])[0]
        spec = self.__add_two_specs(spec,bg_to_add)
        spec /= np.max(spec)

        return spec
        

    def same_class_augmentation(self, spec, label):
        augmented_spec = spec
        # get all samples from the same label id

        same_class_pool = self.inverse_labels[label]
        (lower_to_add, upper_to_add) = self.samples_to_add
        # we add nr_to_add samples of the same class for augmentation
        # nr_to_add is determined as a random number within a given range
        nr_to_add = random.randint(lower_to_add, upper_to_add)
        for i in range(nr_to_add):
            r = random.randint(0, len(same_class_pool) - 1)
            path = same_class_pool[r]
            
            # load the preprocessed sample to augment with
            spec_to_add = self.preprocessor.load_and_preprocess([path])[0]
            augmented_spec = self.__add_two_specs(augmented_spec, spec_to_add)
            augmented_spec /= np.max(augmented_spec)
        
        return augmented_spec


    def __add_two_specs(self, spec1, spec2):
        (f1, t1) = spec1.shape
        (f2, t2) = spec2.shape
        # if the two specs are not of the same size, we artificially repeat the smaller one
        # with our current DataPreparation technique, this should actually never happen
        if t1 < t2:
            nr_repeat = math.ceil(t2 / t1)
            spec1 = np.repeat(spec1, nr_repeat, axis=1)
            spec1 = spec1[:, :t2]
        elif t2 < t1:
            nr_repeat = math.ceil(t1 / t2)
            spec2 = np.repeat(spec2, nr_repeat, axis=1)
            spec2 = spec2[:, :t1]

        return (spec1 + spec2)
        
