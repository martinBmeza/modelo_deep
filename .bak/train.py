"""Contiene el bucle principal de entrenamiento
POR HACER:
    -Asegurar aleatoriedad en la seleccion de datos
    -Sumar validacion
""" 
from model.data_loader import DataGenerator
import glob
import numpy as np
from model.network_architecture import basic_autoencoder


#Defino los parametros del generador de datos 
params = {'dim': 32000, 
          'batch_size': 8,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

# Datasets
audio_list = glob.glob('data/**/*.npy', recursive=True)
audio_numbers = list(range(0, len(audio_list)))
partition = {'train' : audio_numbers[:], 'validation' : audio_numbers[100:]}

# Generators
training_generator = DataGenerator(partition['train'],path = 'data/dummy_data', **params)
validation_generator = DataGenerator(partition['validation'],path = 'data/dummy_data', **params)

print('cantidad de audios de entrenamiento:',len(partition['train']))

modelo = basic_autoencoder()

modelo.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=6,
                    epochs=10)
