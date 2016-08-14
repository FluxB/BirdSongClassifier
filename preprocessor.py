import librosa
from spectrogram import spectrogram
import random
import numpy as np


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
        random_offset = random.uniform(0, self.duration) # augment in time direction!
        y, sr = librosa.load(wav_path, offset=random_offset)
        
        samples_per_slice = sr * self.duration
        nr_slices = y.shape[0] // samples_per_slice
        
        y_slices = np.array_split(y, nr_slices)

        spectrograms = [spectrogram(y_slice) for y_slice in y_slices]

        return spectrograms

    def __preprocess(self, spectrograms):
        preprocessed_spectrograms = []
        for spec in spectrograms:
            preprocessed_spec = spec
            preprocessed_spec = np.log(spec**2)
            preprocessed_spec /= np.max(preprocessed_spec)
            preprocessed_spectrograms.extend(preprocessed_spec)

        return preprocessed_spectrograms
