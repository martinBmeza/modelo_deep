"""Alimentar con datos al bucle de entrenamiento"""
import keras
import numpy as np
import os
import random
import glob
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

        x_clean = np.empty((self.batch_size, 32000))
        x_reverb = np.empty((self.batch_size, 32000))
        Y = np.empty((self.batch_size, 32000))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #Cargo los audios

            [clean, reverb] = np.load(self.path + str(ID) + '.npy')
            [clean, reverb] = np.load(self.path + str(ID) + '.npy')

            x_clean[i] = clean
            x_reverb[i] = reverb
            Y[i] = reverb
        return [x_clean, x_reverb], Y


"""
Explicación util: La clase DataGenerator es basicamente una extensión derivada
de la clase keras.utils.sequence en la cual se trabaja con una secuencia de
baches. Lo principal en esta clase son los métodos "magicos"(polemico)
__getitem__ y __len__.El primero se encarga de definir la cantidad de baches
que va a tener la secuencia, y el segundo recibe el numero de bache actual de
la secuencia (del total de baches, entonces) y devuelve un bache. El resto de
métodos son opcionales y se encuentran bien documentados.
"""
def build_generators(MAIN_PATH, params):
    """
    Crea instancias de la clase DataGenerator (para entrenamiento y valdiacion) a partir de un diccionario donde se determinan los parametros
    del generador de datos, y el path principal.

    PARAMETROS:
        -MAIN_PATH (str) path principal de la carpeta de trabajo
        -params (dict) diccionario con los campos 'dim'(int), 'batch_size'(int), 'shuffle'(bool) para configurar el generador de datos
        -subpath (str) path de la carpeta dentro de data/ de donde tomar los datos. Por defecto esta asignada a 'data_ready' que es donde se encuentran
        los datos procesados. Puede ser util cambiarla a la carpeta data_dummy para trabajar con los datos dummy en ocasiones de debuggeo

    SALIDA:
        -training_generator (DataGenerator) instancia de clase que contiene los datos para pasarse a una instancia de entrenamiento y proveer 
            los datos de entrenamiento al modelo
        -validation_generator (DataGenerator) instancia de clase que contiene los datos para pasarse a una instancia de entrenamiento y proveer 
            los datos de validacion al modelo

        """
    audio_list = glob.glob(params['path']+'/**/*.npy', recursive=True) #PROVISORIO

    #seleccion random de sets
    audio_numbers = list(range(0, len(audio_list)))
    random.shuffle(audio_numbers)
    train_n = int(len(audio_numbers)*0.8) #porcentaje de entrenamiento - validacion
    validation_n = len(audio_numbers) - train_n

    partition = {'train' : audio_numbers[:train_n], 'validation' : audio_numbers[train_n:]}

    # Generators
    training_generator = DataGenerator(partition['train'],  **params)
    validation_generator = DataGenerator(partition['validation'], **params)

    print('Cantidad de datos para entrenamiento:', len(partition['train']))
    print('Cantidad de datos para validacion:', len(partition['validation']))
    return training_generator, validation_generator
