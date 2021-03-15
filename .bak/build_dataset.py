"""formar el dataset dummy y guardarlo en la carpeta correspondiente

POR HACER: 
    -chequear uso de memoria ram, capturar ese posible error
    -trabajar con directorio principal """ 



import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt 

#variables: 
MAIN_PATH = '/home/mrtn/Documents/TESIS/de-reverb/Source/MODELOS/modelo_prueba/' #dell-notebook
LOW = 80 #limite inferior de frecuencias posibles
HIGH = 8000 #limite superior de frecuencias posibles
CANTIDAD = 128

def dummy_seno(duracion, muestras, freq, noisy=False, noisy_coef=0.5):
    """
    Funcion para crear set de datos senoidales, con y sin ruido. Pensado para formar
    dummysets que sirvan de prueba para modelos tipo de-noising
    
    Parameters
    ----------
    duracion : float
        Largo en segundos que va a tener cada seno.
    muestras : int
        Cantidad de muestras que se van a crear en el tiempo de duración.
        relacionar con duracion y fs.
    freq : floar
        Frecuencia de oscilacion del seno.
    noisy : boolean, optional
        Opcion para agregar ruido aleatorio a las señales generadas. por defecto, 
        es falso (senoidales puras). 
    noisy_coef : float, optional
        coeficiente que controla la proporcion de ruido que se va a sumar, 
        asumiendo que la amplitud de la senoidal es de 0.5. El valor por 
        defecto es 0.5.

    Returns
    -------
    seno : np.array()
        señal senoidal resultante.

    """
    t = np.linspace(0, duracion, muestras)
    seno = 0.5 * np.sin(2*np.pi*freq*t)
    if noisy:
        seno = seno + np.random.uniform(low = -noisy_coef, high = noisy_coef, size = seno.shape)
    return seno

def safe_dir(path):
    if os.path.exists(path): 
        print('Guarda: el directorio ya existia, puede contener informacion preexistente')
        return 
    else: 
        os.mkdir(path)
        return 

freqs = np.random.uniform(low=LOW, high=HIGH, size=(CANTIDAD))

train_clean = np.array([dummy_seno(duracion = 2, 
                                   muestras = 32000, 
                                   freq = f) for i,f in enumerate(freqs)])

train_noisy = np.array([dummy_seno(duracion = 2, 
                                   muestras = 32000, 
                                   freq = f, 
                                   noisy = True) for i,f in enumerate(freqs)])

#Formato de guardado [CLEAN, NOISY]
save_path = os.path.join(MAIN_PATH, 'data/dummy_data')
safe_dir(save_path)
save_path = os.path.join(save_path, '')

for i in tqdm(range(CANTIDAD)): 
    file = str(i)+'.npy'
    filename = os.path.join(save_path,file)
    np.save(filename, np.array([train_clean[i,:], train_noisy[i,:]]))

def test_plot():
    clean, noise = np.load(filename)
    plt.subplot(2,1,1)
    plt.plot(clean)
    plt.title('clean')
    plt.subplot(2,1,2)
    plt.title('noisy')
    plt.plot(noise)
    plt.show()
