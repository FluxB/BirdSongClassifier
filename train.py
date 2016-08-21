import sys
from preprocessor import Preprocessor
from augmentation import AugmentTransform
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import random

# main class of programm, starts and organizes the training
class Bird(object):

    # label_path: path to label file which holds the path of all samples and their class id
    def __init__(self, label_path):
        self.label_path = sys.argv[1]
        self.batch_size = 32
        self.queue_size = 2048
        self.nr_epoch = 1
        self.preprocessor = Preprocessor(10)
        self.augmenter = AugmentTransform(20, 10)
        self.inverse_labels = {}

    # loads all sample paths and their id into memory. 
    # builds up an inverse lookup structure to augment
    # samples with the same class. 
    def load_labels(self, label_path):
        f = open(label_path, "r")
        paths = []
        labels = []
        self.inverse_labels = {}
        for line in f:
            line = line.strip()
            (path, label) = line.split(" ")
            paths.append(path)
            labels.append(label)
            self.inverse_labels.setdefault(label, []).append(path)

        self.augmenter.configure_same_class_augmentation(self.inverse_labels,
                                                         self.preprocessor,
                                                         samples_to_add=[1, 2])
        return (paths, labels)


    # loads and randomizes data
    def load_data(self):
        (paths, labels) = self.load_labels(self.label_path)
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
            
            self.queue.put(self.__get_random_training_sample())


    # loads a single new training sample from disc. 
    # preprocesses and augments the training sample.
    def __get_random_training_sample(self):
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
        self.start_queue_filling_process()

        nr_batches = self.nr_files // self.batch_size
        
        for epoch in range(self.nr_epoch):
            for batch_i in range(nr_batches):
                spec_batch = []
                label_batch = []
                for sample in range(self.batch_size):
                    (spec, label) = self.queue.get()
                    spec_batch.append(spec)
                    label_batch.append(label)
                    plt.imshow(spec)
                    plt.show()
                #START TRAINING HERE


if __name__ == "__main__":
    label_path = sys.argv[1]
    bird = Bird(label_path)
    bird.train()
            
