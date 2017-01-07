import sys
from preprocessor import Preprocessor
from augmentation import AugmentTransform
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.optimizers import SGD
import models
from keras.callbacks import ProgbarLogger


# main class of programm, starts and organizes the training
class Bird(object):

    # label_path: path to label file which holds the path of all samples and their class id
    def __init__(self, label_path,label_bg_path, meta_path, output_path):
        self.label_path = label_path
        self.label_bg_path = label_bg_path
        self.meta_path = meta_path
        self.output_path = output_path
        
        self.batch_size = 64
        self.queue_size = 2048

        self.nr_epoch = 10
        self.preprocessor = Preprocessor(10)
        self.augmenter = AugmentTransform(10, 10)
        self.inverse_labels = {}
        self.inverse_labels_bg = {}

        self.train_val_ratio = 0.1
        
    # loads all sample paths and their id into memory. 
    # builds up an inverse lookup structure to augment
    # samples with the same class. 
    def load_labels(self, label_path, label_bg_path):
        f = open(label_path, "r")
        f_bg = open(label_bg_path, "r")
        paths = []
        labels = []

        self.inverse_labels = {}
        self.inverse_labels_bg = {}
        
        for line in f:
            line = line.strip()
            (path, label) = line.split(" ")
            paths.append(path)
            labels.append(label)
            self.inverse_labels.setdefault(label, []).append(path)

        self.nb_species = len(self.inverse_labels)
        print(self.nb_species)
    
        for line in f_bg:
            line = line.strip()
            (path, label) = line.split(" ")
 
            self.inverse_labels_bg.setdefault(label, []).append(path)

        self.augmenter.configure_same_class_augmentation(self.inverse_labels,self.inverse_labels_bg,self.preprocessor,samples_to_add=[1, 2])
        
        return (paths, labels)

    def load_meta_data(self, meta_path):
        with  open(meta_path, "r") as meta:
            first_line=meta.readline()
            first_line = first_line.strip()
            first_line_split = first_line.split(" ")
            nb_f_steps = first_line_split[2]
            nb_t_steps = first_line_split[3]
        self.nb_f_steps = 128 #  int(nb_f_steps)
        self.nb_t_steps = 256 #  int(nb_t_steps)
        
        
    # loads and randomizes data
    def load_data(self):
        (paths, labels) = self.load_labels(self.label_path,self.label_bg_path)
        self.load_meta_data(self.meta_path)
        
        nr_files = len(paths)

        mask = np.arange(nr_files)

        np.random.shuffle(mask)

        train_size = int(nr_files * (1 - self.train_val_ratio))

        paths = np.array(paths)[mask]
        labels = np.array(labels)[mask]

        self.class_weights = {}
        for i in range(self.nb_species):
            weight_mask = labels == str(i)  #  np.equal(labels, i*np.ones(labels.shape))
            nb_class = np.sum(weight_mask)
            if nb_class == 0:
                print("No data for class", str(i))
                continue
            self.class_weights[i] = nr_files/np.sum(weight_mask)

        self.paths = paths[:train_size]
        self.labels = labels[:train_size]
        self.nr_files = train_size

        self.val_paths = paths[train_size:]
        self.val_labels = labels[train_size:]
        self.nr_val_files = (nr_files - train_size) // self.batch_size * self.batch_size


    def train_data_generator(self):
        while True:
            specs = []
            labels = []
            for i in range(self.batch_size):
                (spec, label) = self.get_random_training_sample()
                specs.append(np.array([spec]).transpose((1, 2, 0)))
                labels.append(np.array([label]))

            yield (np.array(specs), np.array(labels))


    def val_data_generator(self):
        specs = []
        labels = []
        for val_path, val_label in zip(self.val_paths, self.val_labels):
            sample = self.preprocessor.load_sample(val_path)
            if np.max(sample[0]) <= 0:
                continue
            spec = self.preprocessor.preprocess(sample)
            # spec = self.augmenter.augment_transform(spec, val_label)
            specs.append(np.array([spec[0]]).transpose((1, 2, 0)))
            labels.append(np.array([val_label]))
            if len(specs) == self.batch_size:
                yield (np.array(specs), np.array(labels))
                specs = []
                labels = []

        if len(specs) > 0:
            yield (np.array(specs), np.array(labels))


    # loads a single new training sample from disc. 
    # preprocesses and augments the training sample.
    def get_random_training_sample(self):
        r = random.randint(0, self.nr_files - 1)
        path = self.paths[r]
        label = self.labels[r]
        sample = self.preprocessor.load_sample(path)
        if np.max(sample[0]) <= 0:
            return self.get_random_training_sample()
        spec = self.preprocessor.preprocess(sample)
        spec = self.augmenter.augment_transform(spec, label)
        return (spec[0], label)

    
    # start training process
    def train(self):
        self.load_data()
        self.model = models.model_paper(self.nb_species,
                                        (self.nb_f_steps, self.nb_t_steps))
        sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        
        self.model.summary()

        progbar = ProgbarLogger()

        history = self.model.fit_generator(self.train_data_generator(), samples_per_epoch=self.nr_files,
                                           nb_epoch=self.nr_epoch, verbose=1, max_q_size=self.batch_size,
                                           validation_data=self.val_data_generator(), nb_val_samples=self.nr_val_files,
                                           nb_worker=1, pickle_safe=True)




if __name__ == "__main__":
    label_path = sys.argv[1]
    label_bg_path = sys.argv[2]
    meta_path = sys.argv[3]
    output_path = sys.argv[4]
    bird = Bird(label_path,label_bg_path,meta_path,output_path)
    bird.train()
            
