# Use this script to load the wavs with librosa and pickle the numpy arrays. 
# This should yield performance gains, as librosa appears slow.
# Usage: python3 data_preparation.py data_path out_path
# where data_path = path to wav files, with each bird class in its own folder (as it is generated by birds.py)
# and out_path = path where the pickles files will be stored.
# additionally the script generates a label, meta and dictionary file.

import os
import sys
import librosa
import numpy as np
import pickle
from spectrogram import spectrogram
import scipy.ndimage.morphology as morph
import matplotlib.pyplot as plt
import multiprocessing as mp

import xml.etree.ElementTree as ET


class DataPreparator(object):

    def __init__(self, data_path, out_path, chunk_duration):
        self.data_path = data_path
        self.out_path = out_path
        self.chunk_duration = chunk_duration
        self.pre_chunk_duration = 1024

        self.label_dict_lock = mp.Lock()
        self.label_lock = mp.Lock()
        self.bg_lock = mp.Lock()

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

    def prepareCLEFTrainingData(self, n_jobs=1):
        f_label = open("labels.txt", "w")
        f_label_bg = open("labels_bg.txt", "w")
        f_meta = open("meta.txt", "w")

        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.isdir(self.out_path + "/bg"):
            os.mkdir(self.out_path + "/bg")

        label_dict = {}
        file_names = os.listdir(self.data_path)

        files_per_process = len(file_names) // n_jobs
        file_names_processes = [file_names[x:x + files_per_process]
                                for x in range(0, len(file_names), files_per_process)]
        processes = []
        for file_names_process in file_names_processes:
            p = mp.Process(target=self.__prepare_processing, args=(file_names_process, label_dict,
                                                                   f_label, f_label_bg, f_meta))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def __prepare_processing(self, file_names, label_dict, f_label, f_label_bg, f_meta):
        data_path_trunk = self.data_path[:-3]

        for file_name in file_names:
            if file_name[0] == ".":
                continue
            tree = ET.parse(self.data_path + "/" + file_name)
            root = tree.getroot()
            class_name = root.find("ClassId").text

            self.label_dict_lock.acquire()
            label_dict[class_name] = label_dict.get(class_name, len(label_dict))
            self.label_dict_lock.release()

            wav = root.find("FileName").text
            metadata = ""
            for child in root:
                text = child.text if child.text is not None else ""
                metadata += text + ","
            self.__process_wav(data_path_trunk + "wav/" + wav,
                               label_dict, class_name, f_label, f_label_bg, f_meta, metadata)

    def __process_wav(self, wav, label_dict, class_name, f_label, f_label_bg, f_meta, additional_meta=""):
        print("Preparing file: ", wav)
        y, sr = librosa.load(wav)
        spec = spectrogram(y)

        spec, bg = self.bg_subtraction(spec)

        chunks = self.make_chunks(spec)

        name, ftype = wav.split(".")
        name = name.split("/")[-1]
        for i, chunk in enumerate(chunks):
            out_fname = str.format("{}/{}_{}.pkl", self.out_path, name, i)
            pickle.dump(chunk, open(out_fname, "wb"), protocol=2)

            self.label_lock.acquire()
            f_label.write(out_fname + " " + str(label_dict[class_name]) + "\n")
            f_meta.write("{} {} {} {} {}\n".format(out_fname, sr, chunk.shape[0],
                                                   chunk.shape[1], additional_meta.encode("utf-8")))
            f_label.flush()
            f_meta.flush()
            self.label_lock.release()

        if bg.shape[1] == 0 or spec.shape[1] == 0:
            return
        bg = np.true_divide(bg, np.max(spec))
        chunks_bg = self.make_chunks(bg)

        for i, chunk in enumerate(chunks_bg):
            out_fname = str.format("{}/bg/{}_bg_{}.pkl", self.out_path, name, i)

            self.bg_lock.acquire()
            f_label_bg.write(out_fname + " " + str(label_dict[class_name]) + "\n")
            f_label_bg.flush()
            self.bg_lock.release()

            pickle.dump(chunk, open(out_fname, "wb"), protocol=2)

    
    def bg_subtraction_chunkwise(self, s):
        split_points = [self.pre_chunk_duration * i for i in range(1, s.shape[1] // self.pre_chunk_duration + 1)]
        s_splits = np.array_split(s, split_points, axis=1)
        fg_list = []
        bg_list = []
        for s_split in s_splits:
            fg, bg = self.bg_subtraction(s_split)
            fg_list.append(fg)
            bg_list.append(bg)
        
        return (np.concatenate(fg_list, axis=1), np.concatenate(bg_list, axis=1))


    def bg_subtraction(self, s):

        mask, vector = self.create_mask(s, threshold=3.0)

        # s_bg = np.array(s,copy=True)

        # signal projection
        # s[np.logical_not(mask)] = 0
        s_signal = s[..., vector]

        mask, vector = self.create_mask(s, threshold=2.5)

        # background projection
        s_bg = s[..., np.logical_not(vector)]

        return s_signal, s_bg

    def create_mask(self, s, threshold=3.0):  # Creates mask to kill background noise and quiet times
        nb_f, nb_t = len(s[:, 0]), len(s[0, :])

        # Set to zero values smaller than 3 times fixed-t-median or smaller 3 times fixed-f-median
        m_f = np.tile(np.median(s, axis=0), (nb_f, 1))
        m_t = np.transpose(np.tile(np.transpose(np.median(s, axis=1)), (nb_t, 1)))

        mask_inv = np.logical_or((s < threshold * m_f), (s < threshold * m_t))
        mask = np.logical_not(mask_inv)

        # morphological trafos: dilation and erosion
        mask = morph.binary_erosion(mask, structure=np.ones((4, 4)), iterations=1)
        mask = morph.binary_dilation(mask, structure=np.ones((4, 4)), iterations=1)

        # make continous twittering by use of killig vector :)
        vector = np.sum(mask, axis=0) > 0
        vector = morph.binary_dilation(vector, structure=np.ones((4)), iterations=2)

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
    preparator.prepareCLEFTrainingData(n_jobs=8)
