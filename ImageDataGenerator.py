import numpy as np
from tensorflow import keras
import cv2
import random

from Configuration import Configuration

class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_names, labels, configuration : Configuration):
        self.dimension = configuration.Shape
        self.labels = labels
        self.image_names = image_names
        self.batch_size = configuration.BatchSize
        self.indexes = np.arange(len(self.image_names))
        np.random.shuffle(self.indexes)

    def __data_generation(self, image_names):
            
        x_images = []
        y_labels = [] 

        # Generate data
        for index, image_name in enumerate(image_names):

            image = np.float32(cv2.imread(image_name, cv2.IMREAD_COLOR))

            if (image is not None):
                image = cv2.resize(image, dsize=(self.dimension[1], self.dimension[0]), interpolation=cv2.INTER_CUBIC)
                x_images.append(image)
                y_labels.append(self.labels[image_name])

        X = np.asarray(x_images)   
        y = np.asarray(y_labels)   

        return X, y

    def __len__(self):

        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_names) / self.batch_size))

    def __getitem__(self, index, shouldReturnPaths = False):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        image_names_temp = [self.image_names[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(image_names_temp)

        if (shouldReturnPaths):
            return X, y, image_names_temp
        
        return X, y