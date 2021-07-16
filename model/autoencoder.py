import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
eps = np.finfo(float).eps

#falta hacer el acondicionamiento afuera
def basic_autoencoder():

    entrada = tfkl.Input((256,256,1), name = 'Clean')

    #ENCODER
    enc = tfkl.Conv2D(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET1')(entrada)
    enc = tfkl.Conv2D(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET2')(enc)
    enc = tfkl.Conv2D(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET3')(enc)
    enc = tfkl.Conv2D(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET4')(enc)
    enc = tfkl.Conv2D(64, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET5')(enc)
    enc = tfkl.Conv2D(64, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET6')(enc)

    #DECODER
    dec = tfkl.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET7')(enc)
    dec = tfkl.Conv2DTranspose(64, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET8')(dec)
    dec = tfkl.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET9')(dec)
    dec = tfkl.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET10')(dec)
    dec = tfkl.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET11')(dec)
    dec = tfkl.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET12')(dec)
    salida = tfkl.Conv2DTranspose(1, kernel_size=(3,3), padding='SAME', activation='relu', name='salida')(dec)


    modelo = tf.keras.Model(inputs=[entrada],outputs=[salida])
    modelo.compile(optimizer = 'adam', loss = 'mse')
    return modelo 

if __name__ == '__main__':
    modelo = basic_autoencoder()
    modelo.summary()


