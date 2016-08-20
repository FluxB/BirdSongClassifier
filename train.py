import sys
from preprocessor import Preprocessor
from augmentation import AugmentTransform
import numpy as np
import matplotlib.pyplot as plt


def load_labels(label_path):
    f = open(label_path, "r")
    paths = []
    labels = []
    for line in f:
        (path, label) = line.split(" ")
        paths.append(path)
        labels.append(label)

    return (paths, labels)


label_path = sys.argv[1]
batch_size = 2


(paths, labels) = load_labels(label_path)
nr_files = len(paths)

mask = np.arange(nr_files)

np.random.shuffle(mask)

paths = np.array(paths)[mask]
labels = np.array(labels)[mask]

nr_of_batches = nr_files // batch_size

paths_batches = np.array_split(paths, nr_of_batches)
labels_batches = np.array_split(labels, nr_of_batches)

preprocessor = Preprocessor(10)
augment = AugmentTransform(20)

for path_batch, label_batch in zip(paths_batches, labels_batches):
    spectrograms_batch = preprocessor.load_and_preprocess(path_batch)
    spectrograms_batch = augment.augment_transform(spectrograms_batch)
    # for spec in spectrograms_batch:
    #    print(spec.shape)
    #    plt.imshow(spec)
    #    plt.show()
    
