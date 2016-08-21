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

    def load_and_preprocess(self, batch_wav_paths):
        preprocessed_spectrograms = []
        for wav_path in batch_wav_paths:
            wav_path = wav_path.strip()
            specs = self.load_sample(wav_path)
            preprocessed_spectrograms.extend(self.preprocess(specs))

        return preprocessed_spectrograms


    def load_npy(self,np_path):
        print("Loading npy file: ", np_path)
        y = np.load(np_path)
        return spectrogram(y)
    
    def load_sample(self, sample_path):
        print("Loading wav file: ", sample_path)
        y = np.load(sample_path)
        sr = 22050
    
        samples_per_slice = sr * self.duration

        y_slice = y[0:samples_per_slice]
        
        spectrograms = [spectrogram(y_slice)]  # keep list formulation, if we want to use different preprocessing scheme

        return spectrograms

    def preprocess(self, spectrograms):
        preprocessed_spectrograms = []
        for spec in spectrograms:
            preprocessed_spec = spec
            preprocessed_spec = self.bg_sub(preprocessed_spec)
            preprocessed_spec = np.log(spec**2)
            preprocessed_spec /= np.max(preprocessed_spec)
            preprocessed_spectrograms.extend([preprocessed_spec])

        return preprocessed_spectrograms

    def bg_subtraction(self,s):

        mask,vector = self.create_mask(s)

        #signal
        s_signal = s
        s_signal[np.logical_not(mask)] = 0
        s_signal = s_signal[...,vector]
        return s_signal
        #background
    
        s_bg = s[...,np.logical_not(vector)]
        
    def create_mask(self,s): # Creates mask to kill background noise and quite times
        nb_f, nb_t = len(s[:,0]), len(s[0,:])
        
        # normalize naively
        sn=np.true_divide(s, np.max(s))
        
        
        # to Set zero values smaller than 3 times fixed-t-median or smaller 3 times fixed-f-median
        m_f = np.tile(np.median(sn,axis=0),(nb_f,1))
        m_t = np.transpose(np.tile(np.transpose(np.median(sn,axis=1)),(nb_t,1)))

        
       
        mask_inv = np.logical_or((sn < 3 * m_f),(sn < 3 * m_t))
        mask = np.logical_not(mask_inv)
        
        #morphological trafos: dilation and erosion
        mask = morph.binary_erosion(mask,structure=np.ones((4,4)),iterations=1)
        mask = morph.binary_dilation(mask,structure=np.ones((4,4)),iterations=1)

        #make continous twittering by use of killig vector :)
        vector = np.sum(mask,axis=0)>0
        vector = morph.binary_dilation(vector,structure=np.ones((1)),iterations=1)
        
        return mask, vector
