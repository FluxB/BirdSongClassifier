import librosa
from spectrogram import spectrogram
import random
import numpy as np
import time


class Preprocessor(object):

    def __init__(self, duration):
        self.duration = duration

    def load_and_preprocess(self, batch_wav_paths):
        preprocessed_spectrograms = []
        for wav_path in batch_wav_paths:
            wav_path = wav_path.strip()
            specs = self.__load_wav(wav_path)
            preprocessed_spectrograms.extend(self.__preprocess(specs))

        return preprocessed_spectrograms
            

    def __load_wav(self, wav_path):
        print("Loading wav file: ", wav_path)
        y, sr = librosa.load(wav_path)
    
        samples_per_slice = sr * self.duration
        random_offset = random.randint(0, y.shape[0] - samples_per_slice)

        y_slice = y[random_offset : (random_offset + samples_per_slice)]
        
        spectrograms = [spectrogram(y_slice)]  # keep list formulation, if we want to use different preprocessing scheme


        return spectrograms

    def __preprocess(self, spectrograms):
        preprocessed_spectrograms = []
        for spec in spectrograms:
            preprocessed_spec = spec
            preprocessed_spec = np.log(spec**2)
            preprocessed_spec /= np.max(preprocessed_spec)
            preprocessed_spectrograms.extend(preprocessed_spec)

        return preprocessed_spectrograms
