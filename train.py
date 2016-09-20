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
from keras.optimizers import SGD
import models


# main class of programm, starts and organizes the training
class Bird(object):

    # label_path: path to label file which holds the path of all samples and their class id
    def __init__(self, label_path,label_bg_path, meta_path, output_path):
        self.label_path = label_path
        self.label_bg_path = label_bg_path
        self.meta_path = meta_path
        self.output_path = output_path
        
        self.batch_size = 16
        self.queue_size = 2048

        self.nr_epoch = 1
        self.preprocessor = Preprocessor(10)
        self.augmenter = AugmentTransform(10, 10)
        self.inverse_labels = {}
        self.inverse_labels_bg = {}
        
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
            (path, sr, nb_f_steps, nb_t_steps) = first_line.split(" ")
        self.nb_f_steps = int(nb_f_steps)
        self.nb_t_steps = int(nb_t_steps)
        
        
    # loads and randomizes data
    def load_data(self):
        (paths, labels) = self.load_labels(self.label_path,self.label_bg_path)
        self.load_meta_data(self.meta_path)
        
        self.nr_files = len(paths)

        mask = np.arange(self.nr_files)

        np.random.shuffle(mask)

        self.paths = np.array(paths)[mask]
        self.labels = np.array(labels)[mask]


    # kicksoff second thread that asynchronously fills data into memory
    def start_queue_filling_process(self):
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.process = mp.Process(target=self.__fill_queue)
        self.process.start()

    
    def __fill_queue(self):
        while True:
            while self.queue.full():
                time.sleep(1)
            
            self.queue.put(self.get_random_training_sample())


    # loads a single new training sample from disc. 
    # preprocesses and augments the training sample.
    def get_random_training_sample(self):
        r = random.randint(0, self.nr_files - 1)
        path = self.paths[r]
        label = self.labels[r]
        sample = self.preprocessor.load_sample(path)
        spec = self.preprocessor.preprocess(sample)
        spec = self.augmenter.augment_transform(spec, label)
        return (spec[0], label)

    
    # start training process
    def train(self):
        self.load_data()
        model=models.model_paper(self.nb_species)
        self.start_queue_filling_process()
        nr_batches = self.nr_files // self.batch_size
        
        for epoch in range(self.nr_epoch):
            for batch_i in range(nr_batches):
                spec_batch = []
                label_batch = []
                for sample in range(self.batch_size):
                    (spec, label) = self.queue.get()
                    spec_batch.append([spec])
                    label_batch.append(label)
                    #plt.imshow(spec)
                    #plt.show()e
                
                #model.fit(spec_batch,label_batch, batch_size=self.batch_size, verbose=1, nb_epoch=1)

        #model.save(self.output_path)



if __name__ == "__main__":
    label_path = sys.argv[1]
    label_bg_path = sys.argv[2]
    meta_path = sys.argv[3]
    output_path = sys.argv[4]
    bird = Bird(label_path,label_bg_path,meta_path,output_path)
    bird.train()
            
