"""
Archivo para generar los datos de prueba para el modelo. La idea es generar señales senoidales puras
y señales senoidales con ruido agregado. Cada tipo de señal se guarda luego en las carpetas "clean" y
"noisy" respectivamente.
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os, sys

#Parametros de Generacion
cantidad = 1000
duracion = 2 #segundos
fs=16000

def safe_dir(path):
        if os.path.exists(path):
            print('Guarda: el directorio ya existia, puede contener informacion preexistente')
            return path
        else:
            os.mkdir(path)
            return path


def generar_seno(frecuencia, duracion, fs, amplitud=0.7):
    """Generacion de señales senoidales. Duracion en segundos, frecuencia en Hz
    """
    tiempo = np.linspace(0, duracion ,(fs*duracion))
    seno = amplitud*np.sin(2*np.pi*frecuencia*tiempo)
    return seno

#plt.plot(generar_seno(440))
#plt.show()


#sf.write("prueba.wav", generar_seno(440), 16000)

clean = [generar_seno(i, duracion, fs) for i in np.random.uniform(200, 2000, size=(cantidad,))]
noisy = [(clean[j]+np.random.uniform(-0.3, 0.3, size=(duracion*fs,))) for j in range(len(clean)) ]



safe_dir('clean')
safe_dir('noisy')


if len(clean)==len(noisy):
       print("se generaron conjuntos homogeneos de {} señales".format(len(clean)))
else:
    raise Exception("ERROR: Las señales no se estan generando en igual cantidad")
for i in range(cantidad):
    filename = 'data_dummy/'+str(i)+'.npy'
    np.save(filename, [clean[i], noisy[i]])
