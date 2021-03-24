"""Alimentar con datos al bucle de entrenamiento"""
import keras
import numpy as np
import os
MAIN_PATH = '/home/martin/Documents/modelo_deep'

class DataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''

    def __init__(self, list_IDs, path, labels = None, batch_size=32, dim=(513,33), n_channels=1, n_classes=None, shuffle=True):
        '''Initialization'''
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
        '''Indica el numero de baches por epoca. Esta definido como
        la cantidad de elementos con distintos IDs dividido el tamaño de baches buscado.
        Este valor se redondea para obtener numeros enteros.'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        '''Genera un bacht de datos. Recibe los indices que tiene que seleccionar, y
        devuelve los vectores que conforman el batch de datos'''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        '''Actualiza los indices a utilizar luego de cada epoca.
        Hace una selección random si shuffle==True. De lo contrario,
        los indices se mantienen iguales y el orden de entrenamiento
        es siempre el mismo'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Genera los datos para un bache en particular. De este metodo depende el
        formato de salida de los tensores que forman el bache, como tambien la
        ubicacion de los datos.Generates data containing batch_size samples'''
        # Initialization
        X = np.empty((self.batch_size, 2, 32000))
        Y = np.empty((self.batch_size, 32000))
        path = os.path.join(MAIN_PATH, self.path,"") #No tiene sentido que esto se repita en cada epoca.
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Cargo los audios
            [x, y] = np.load(path + str(ID) + '.npy')
            X[i,:] = x,y
            Y[i] = y
        return [X[:,0],X[:,1]], Y
"""
Explicación util: La clase DataGenerator es basicamente una extensión derivada
de la clase keras.utils.sequence en la cual se trabaja con una secuencia de
baches. Lo principal en esta clase son los métodos "magicos"(polemico)
__getitem__ y __len__.El primero se encarga de definir la cantidad de baches
que va a tener la secuencia, y el segundo recibe el numero de bache actual de
la secuencia (del total de baches, entonces) y devuelve un bache. El resto de
métodos son opcionales y se encuentran bien documentados.
"""
