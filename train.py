"""
Bucle de entrenamiento
"""
import numpy as np
import tensorflow as tf
import sys
MAIN_PATH="/home/martin/Documents/modelo_deep"
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader import build_generators
path = MAIN_PATH + '/data/data_dummy/'
params = {'dim': (257,256), 'batch_size': 16, 'shuffle': True, 'path' : path}
training_generator, validation_generator = build_generators(MAIN_PATH, params)

def dummy_seno(duracion, muestras, freq, noisy=False, noisy_coef=0.5):
    t = np.linspace(0, duracion, muestras)
    seno = 0.5 * np.sin(2*np.pi*freq*t)
    if noisy:
        seno = seno + np.random.uniform(low = -noisy_coef, high = noisy_coef, size = seno.shape)
    return seno

freqs = np.random.uniform(low=80, high=8000, size=(1024))
#freqs = 440 * np.ones(128) #para probar cosas 

train_clean = np.array([dummy_seno(duracion = 2, muestras = 32000, freq = f) for i,f in enumerate(freqs)])
train_noisy = np.array([dummy_seno(duracion = 2, muestras = 32000, freq = f, noisy = True) for i,f in enumerate(freqs)])



#defino el modelo
from model.network_architecture import deep_seno, mean_loss
modelo = deep_seno()
modelo.summary()

#defino optimizador y loss
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=mean_loss)

#callbacks
#--------------------


#entrenando g
history = modelo.fit(training_generator, epochs= 10)
#modelo.fit([train_noisy,train_clean],[train_noisy], batch_size=64, epochs=20)
