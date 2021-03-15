"""
Archivo para generar los datos de prueba para el modelo. La idea es generar señales senoidales puras
y señales senoidales con ruido agregado. Cada tipo de señal se guarda luego en las carpetas "clean" y
"noisy" respectivamente.
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys

#Parametros de Generacion
cantidad = 1000
duracion = 1
fs=16000

def generar_seno(frecuencia, duracion, fs, amplitud=0.7):
    """Generacion de señales senoidales. Duracion en segundos, frecuencia en Hz
    """
    tiempo = np.linspace(0,1,(fs*1)+1)
    seno = amplitud*np.sin(2*np.pi*frecuencia*tiempo)
    return seno

#plt.plot(generar_seno(440))
#plt.show()


#sf.write("prueba.wav", generar_seno(440), 16000)

clean = [generar_seno(i, duracion, fs) for i in np.random.uniform(200, 2000, size=(cantidad,))]
noisy = [(clean[j]+np.random.uniform(-0.3, 0.3, size=(duracion*fs+1,))) for j in range(len(clean)) ]



if len(clean)==len(noisy):
       print("se generaron conjuntos homogeneos de {} señales".format(len(clean)))
else:
    raise Exception("ERROR: Las señales no se estan generando en igual cantidad")
for i in range(cantidad):
    nombre_clean = "clean/senoidal_"+str(i)+".flac"
    nombre_noisy = "noisy/senoidal_"+str(i)+".flac"
    sf.write(nombre_clean, clean[i], fs)
    sf.write(nombre_noisy, noisy[i], fs)

#print("largo de clean: "+str(len(clean)))
#print("largo de noisy: "+str(len(noisy)))

#
#plt.plot(clean[0][:4000])
#plt.plot(noisy[0][:4000])
#plt.show()


