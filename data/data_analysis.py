import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

eps = np.finfo(float).eps #resolucion de punto flotante
#Traigo un ejemplo de audio clean y audio noisy

clean =2* np.sin(2*np.pi*1000*(np.linspace(0,1,16000)))
#[clean, fs] = sf.read('clean/senoidal_98.flac') #tupla (vector de samples "ndarray", fs "int")
[noisy, fs] = sf.read('noisy/senoidal_98.flac')
if len(clean)==len(noisy):
    print('Cantidad de muestras: {}'.format(len(clean)))
else:
    print('cantidad de muestras desiguales entre grupos clean y noisy')

#Calculo una fft para ver dimensiones y formatos de las distintas librerias

#scipy 

from scipy import signal
import matplotlib.cm as cm 
from matplotlib.colors import Normalize


freq, time, Z = signal.stft(clean, window='hann', nperseg=512, noverlap=256)
freq, time, Z_n = signal.stft(noisy, window='hann', nperseg=512, noverlap=256)

def scipy_plot(Z, Z_n, param_dict,):
    """Graficador para stft de scipy. Grafica usando pcolormesh de pyplot.

    Argumentos:
    ----------
    Z, Z_n : ndarray matrix
        Matriz de numeros que se quiere plotear

    param_dict : Dict
        Diccionario con los kwargs que se le pasa a pcolormesh

    Salida:
    ------
    """
    cmap = cm.get_cmap('viridis')
    normalizer = Normalize(np.min([np.min(Z),np.min(Z_n)]),np.max([np.max(Z),np.max(Z_n)]))
    im = cm.ScalarMappable(norm=normalizer)

    y = np.linspace(1, 8000, num=257)
    x = np.linspace(0, len(clean)/fs,num=64)
    X,Y = np.meshgrid(x,y)

    fig, (ax_clean, ax_noisy) = plt.subplots(2,1)

    ax_clean.set_yscale('log')
    ax_clean.set_ylim([40,8000])
    ax_clean.set_yticks([63,125, 250, 500, 1000, 2000, 4000, 8000])
    ax_clean.set_yticklabels([63,125, 250, 500, 1000, 2000, 4000, 8000])
    ax_clean.set_xlabel("Tiempo [s]")
    ax_clean.set_ylabel("Frecuencia [Hz]")
    ax_clean.set_title('Clean', size=20)
    ax_clean.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, norm=normalizer, **param_dict)

    ax_noisy.set_yscale('log')
    ax_noisy.set_ylim([40,8000])
    ax_noisy.set_yticks([63,125, 250, 500, 1000, 2000, 4000, 8000])
    ax_noisy.set_yticklabels([63,125, 250, 500, 1000, 2000, 4000, 8000])
    ax_noisy.set_xlabel("Tiempo [s]")
    ax_noisy.set_ylabel("Frecuencia [Hz]")
    ax_noisy.set_title('Noisy', size=20)
    ax_noisy.pcolormesh(X, Y, Z_n, shading='auto', cmap=cmap, norm=normalizer, **param_dict)

    fig.colorbar(im, ax=[ax_clean, ax_noisy])
    plt.show()
    return 

scipy_plot(abs(Z), abs(Z_n), {})

sf.write("clean.wav", clean, fs)
sf.write("noisy.wav", noisy, fs)


#librosa

