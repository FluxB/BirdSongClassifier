# Use this script to load the wavs with librosa and pickle the numpy arrays. this should yield performance gains, as librosa appears slow.
# Usage: python3 data_preparation.py data_path out_path
# where data_path = path to wav files, with each bird class in its own folder (as it is generated by birds.py)
# and out_path = path where the pickles files will be stored. 
# additionally the script generates a label, meta and dictionary file.

import os
import sys
import librosa
import numpy as np
import pickle

data_path = sys.argv[1]
out_path = sys.argv[2]

f_label = open("labels.txt", "w")
f_meta = open("meta.txt", "w")

if not os.path.isdir(out_path):
    os.mkdir(out_path)

folder_names = os.listdir(data_path)
label_dict = {}

for folder in folder_names:
    if not os.path.isdir(data_path + folder) or folder[0] == ".":
        continue
    
    print("Preparing folder: ", folder)
    wavs = os.listdir(data_path + folder)

    label_dict[folder] = len(label_dict)

    for wav in wavs:
        if not os.path.isfile(data_path + folder + "/" + wav) or wav[0] == ".":
            continue
        print("Preparing file: ", wav)
        y, sr = librosa.load(data_path + folder + "/" + wav)
        name, ftype = wav.split(".")
        out_fname = out_path + name + ".npy"
        np.save(out_fname, y)
        f_label.write(out_fname + " " + str(label_dict[folder]) + "\n")
        f_meta.write(out_fname + "," + str(sr) + "," + str(y.shape[0]) + "\n")

pickle.dump(label_dict, open("label_dict.pickle", "wb"))
