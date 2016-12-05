# Use this script to load the wavs with librosa and pickle the numpy arrays. This should yield performance gains, as librosa appears slow.
# Usage: python3 data_preparation.py data_path out_path
# where data_path = path to wav files, with each bird class in its own folder (as it is generated by birds.py)
# and out_path = path where the pickles files will be stored. 
# additionally the script generates a label, meta and dictionary file.

import os
import sys
import librosa
import numpy as np
import pickle
import hickle
from spectrogram import spectrogram
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET


class DataPreparator(object):

    def __init__(self, data_path, out_path, chunk_duration):
        self.data_path = data_path
        self.out_path = out_path
        self.chunk_duration = chunk_duration

    def prepareTrainingData(self):
        f_label = open("labels.txt", "w")
        f_label_bg = open("labels_bg.txt", "w")
        f_meta = open("meta.txt", "w")

        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.isdir(self.out_path + "/bg"):
            os.mkdir(self.out_path + "/bg")

        folder_names = os.listdir(self.data_path)
        label_dict = {}

        for folder in folder_names:
            if not os.path.isdir(self.data_path + folder) or folder[0] == ".":
                continue

            print("Preparing folder: ", folder)
            wavs = os.listdir(self.data_path + folder)

            label_dict[folder] = len(label_dict)

            for wav in wavs:
                if not os.path.isfile(self.data_path + folder + "/" + wav) or wav[0] == ".":
                    continue
                self.__process_wav(wav, label_dict, folder, f_label, f_label_bg, f_meta)

        pickle.dump(label_dict, open("label_dict.pickle", "wb"))

    def prepareCLEFTrainingData(self):
        f_label = open("labels.txt", "w")
        f_label_bg = open("labels_bg.txt", "w")
        f_meta = open("meta.txt", "w")

        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.isdir(self.out_path + "/bg"):
            os.mkdir(self.out_path + "/bg")

        label_dict = {}
        file_names = os.listdir(self.data_path)

        data_path_trunk = self.data_path[:-3]

        for file_name in file_names:
            if file_name[0] == ".":
                continue
            tree = ET.parse(self.data_path + "/" + file_name)
            root = tree.getroot()
            class_name = root.find("ClassId").text
            wav = root.find("FileName").text
            metadata = ""
            for child in root:
                metadata += child.text + ","
            self.__process_wav(data_path_trunk + wav, label_dict, class_name, f_label, f_label_bg, f_meta, metadata)

    def __process_wav(self, wav, label_dict, class_name, f_label, f_label_bg, f_meta, additional_meta=""):
        print("Preparing file: ", wav)
        y, sr = librosa.load(self.data_path + class_name + "/" + wav)
        spec = spectrogram(y)

        # normalize naively
        specb= np.true_divide(spec, np.max(spec))

        spec, bg = self.bg_subtraction(spec)

        chunks = self.make_chunks(spec)

        name, ftype = wav.split(".")
        for i, chunk in enumerate(chunks):
            out_fname = str.format("{}/{}_{}.hkl", self.out_path, name, i)
            hickle.dump(chunk, open(out_fname, "w"))
            f_label.write(out_fname + " " + str(label_dict[class_name]) + "\n")
            f_meta.write("{} {} {} {} {}\n".format(out_fname, sr, chunk.shape[0], chunk.shape[1], additional_meta))

        if bg.shape[1] == 0 or spec.shape[1] == 0:
            return
        bg = np.true_divide(bg, np.max(spec))
        chunks_bg = self.make_chunks(bg)

        for i, chunk in enumerate(chunks_bg):
            out_fname = str.format("{}/{}_bg_{}.hkl", self.out_path, name, i)
            f_label_bg.write(out_fname + " " + str(label_dict[class_name]) + "\n")
            hickle.dump(chunk, open(out_fname, "w"))

    def bg_subtraction(self, s):

        mask, vector = self.create_mask(s)

        # s_bg = np.array(s,copy=True)

        # signal projection
        # s[np.logical_not(mask)] = 0
        s_signal = s[..., vector]

        # background projection
        s_bg = s[..., np.logical_not(vector)]

        return s_signal, s_bg

    def create_mask(self, s):  # Creates mask to kill background noise and quiet times
        nb_f, nb_t = len(s[:, 0]), len(s[0, :])

        # Set to zero values smaller than 3 times fixed-t-median or smaller 3 times fixed-f-median
        m_f = np.tile(np.median(s, axis=0), (nb_f, 1))
        m_t = np.transpose(np.tile(np.transpose(np.median(s, axis=1)), (nb_t, 1)))

        mask_inv = np.logical_or((s < 3 * m_f), (s < 3 * m_t))
        mask = np.logical_not(mask_inv)

        # morphological trafos: dilation and erosion
        mask = morph.binary_erosion(mask, structure=np.ones((4, 4)), iterations=1)
        mask = morph.binary_dilation(mask, structure=np.ones((4, 4)), iterations=1)

        # make continous twittering by use of killig vector :)
        vector = np.sum(mask, axis=0) > 0
        vector = morph.binary_dilation(vector, structure=np.ones((2)), iterations=1)

        return mask, vector

    def make_chunks(self, spec):
        nr_chunks = spec.shape[1] // self.chunk_duration
        split_points = [i * self.chunk_duration for i in range(1, nr_chunks + 1)]
        spec_split = np.array_split(spec, split_points, axis=1)
        chunks = []
        for chunk in spec_split:
            (chunk_frequ, chunk_length) = chunk.shape
            if chunk_length < self.chunk_duration:
                padded_chunk = np.zeros((chunk_frequ, self.chunk_duration))
                padded_chunk[:, :chunk_length] = chunk
                chunk = padded_chunk
            chunks.append(chunk)

        return chunks


if __name__ == "__main__":
    data_path = sys.argv[1]
    out_path = sys.argv[2]

    preparator = DataPreparator(data_path, out_path, 512)
    preparator.prepareCLEFTrainingData()
