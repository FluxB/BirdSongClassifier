import unittest
import numpy as np
from spectrogram import plot_spect
from data_preparation import DataPreparator
from preprocessor import Preprocessor
from train import Bird
import librosa


class Test_bg_subtraction(unittest.TestCase):

    def _test_bg_subtraction(self):
        p=Preprocessor(10)
        s=p.load_npy('./test.npy')
        generator = DataPreparator("", "", 512)
        mask, vector = generator.create_mask(s)
        self.assertGreater(np.sum(vector), 120)

    def _test_bg_subtraction2(self):
        p = Preprocessor(10)
        s = p.load_npy('./test.npy')
        generator = DataPreparator("", "", 512)
        samples1=len(s[0,:])
        
        snew,sbg = generator.bg_subtraction(s)
        samples2=len(snew[0,:])

        self.assertGreater(samples1, samples2)
        #plot_spect([snew,p.load_npy('./test.npy'),sbg])


    def test_augment(self):
        bird = Bird("./labels.txt","./labels_bg.txt","./meta.txt","./")
        bird.load_data()
        (spec, label)=bird.get_random_training_sample()
        #plot_spect([spec])

    def test_train(self):
        bird = Bird("./labels.txt","./labels_bg.txt","./meta.txt","./")
        bird.train()
        (spec, label)=bird.get_random_training_sample()
        #plot_spect([spec])

        
if __name__ == '__main__':
    unittest.main()
