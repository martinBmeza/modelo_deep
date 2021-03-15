"""Alimentar con datos al bucle de entrenamiento"""
import keras 
import numpy as np
import os
MAIN_PATH = '/home/mrtn/Documents/TESIS/de-reverb/Source/MODELOS/modelo_prueba'

class DataGenerator(keras.utils.Sequence):
    
    'Generates data for Keras'
    def __init__(self, list_IDs, path, labels = None, batch_size=32, dim=(513,33), n_channels=1, n_classes=None, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, 2, 32000))
        Y = np.empty((self.batch_size, 32000))
        path = os.path.join(MAIN_PATH, self.path,"")
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Cargo los audios
            [x, y] = np.load(path + str(ID) + '.npy')
            X[i,:] = x,y
            Y[i] = y
        return [X[:,0],X[:,1]], Y

