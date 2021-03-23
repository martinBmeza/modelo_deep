import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

#Traigo un ejemplo de audio clean y audio noisy

clean = sf.read('clean/senoidal_0.flac') #tupla (vector de samples "ndarray", fs "int")
noisy = sf.read('noisy/senoidal_0.flac')
if len(clean[0])==len(noisy[0]):
    print('Cantidad de muestras: {}'.format(len(clean[0])))
else:
    print('cantidad de muestras desiguales entre grupos clean y noisy')

