import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tqdm
import os, sys

def safe_dir(path):
    if os.path.exists(path):
        print('Guarda: el directorio ya existia, puede contener informacion preexistente')
    else:
        os.mkdir(path)
    return path


def generar_seno(frecuencia, duracion, fs, amplitud=0.7):
    """Generacion de señales senoidales. Duracion en segundos, frecuencia en Hz
    """
    tiempo = np.linspace(0, duracion ,(fs*duracion))
    seno = amplitud*np.sin(2*np.pi*frecuencia*tiempo)
    return seno

if __name__ == '__main__':
    
    #Parametros de Generacion
    cantidad = 1000
    duracion = 2 #segundos
    fs=16000

    clean = [generar_seno(i, duracion, fs) for i in np.random.uniform(200, 2000, size=(cantidad,))]
    noisy = [(clean[j]+np.random.uniform(-0.3, 0.3, size=(duracion*fs,))) for j in range(len(clean)) ]

    save_path = safe_dir('data_dummy')
    for i in tqdm.tqdm(range(cantidad)):
        filename = os.path.join(save_path, str(i) + '.npy')
        np.save(filename, [clean[i], noisy[i]])
