from __future__ import division
import librosa
from spectrogram import spectrogram
import random
import numpy as np
import time
import scipy.ndimage.morphology as morph

class Preprocessor(object):

    def __init__(self, duration):
        self.duration = duration
        

    def load_and_preprocess(self, batch_sample_paths):
        preprocessed_spectrograms = []
        for sample_path in batch_sample_paths:
            sample_path = sample_path.strip()
            specs = self.load_sample(sample_path)
            preprocessed_spectrograms.extend(self.preprocess(specs))

        return preprocessed_spectrograms


    def load_npy(self,np_path):
        #print("Loading npy file: ", np_path)
        y = np.load(np_path)
        return spectrogram(y)
    
    def load_sample(self, sample_path):
        #print("Loading spec: ", sample_path)
        spec = np.load(sample_path)
    
        spectrograms = [spec]  # keep list formulation, if we want to use different preprocessing scheme

        return spectrograms


    def preprocess(self, spectrograms):
        preprocessed_spectrograms = []
        for spec in spectrograms:
            preprocessed_spec = spec
            # preprocessed_spec = np.log(spec**2)
            preprocessed_spec /= np.max(preprocessed_spec)
            preprocessed_spectrograms.extend([preprocessed_spec])

        return preprocessed_spectrograms

