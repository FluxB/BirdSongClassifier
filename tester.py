import unittest
import numpy as np
from spectrogram import plot_spect
from data_preparation import DataPreparator
from preprocessor import Preprocessor


class Test_bg_subtraction(unittest.TestCase):

    def test_bg_subtraction(self):
        p=Preprocessor(10)
        s=p.load_npy('./test.npy')
        generator = DataPreparator("", "", 512)
        mask, vector = generator.create_mask(s)
        self.assertGreater(np.sum(vector), 120)

    def test_bg_subtraction2(self):
        p = Preprocessor(10)
        s = p.load_npy('./test.npy')
        generator = DataPreparator("", "", 512)
        samples1=len(s[0,:])
        
        snew = generator.bg_subtraction(s)
        samples2=len(snew[0,:])

        self.assertGreater(samples1, samples2)
        #plot_spect([snew,p.load_npy('./test.npy')])


    

if __name__ == '__main__':
    unittest.main()
