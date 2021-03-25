""" Estructura de red a entrenar
"""

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from keras import layers, models, losses
import numpy as np
from model.my_layers import * 
eps = np.finfo(float).eps

def deep_seno():

    x = tfkl.Input((32000), name = 'Clean')
    y = tfkl.Input((32000), name = 'Noisy')

   #GENERACION DE ESPECTROS --- Pipeline previo --------------------------------
    #phase_x = Spectrogram(1024,512, calculate = 'phase', name='PHASE_X')(x)
    spec_x = Spectrogram(1024,512,name='STFT_X')(x)
    spec_x = tf.math.log(spec_x+eps)
    spec_x = TranslateRange(original_range=[-5, 5],target_range=[0,1.0])(spec_x)
    spec_x = tf.expand_dims(spec_x,axis=-1, name= 'expand_X')
    spec_x = tfkl.Cropping2D(((0,1),(0,1)), name = 'Crop_X')(spec_x)

    spec_y = Spectrogram(1024,512,name='STFT_Y')(y)
    #spec_y = tf.math.log(spec_y+eps)
    #spec_y = TranslateRange(original_range=[-5,5],target_range=[0,1.0])(spec_y)
    spec_y = tf.expand_dims(spec_y,axis=-1, name='expand_Y')
    #spec_y = tfkl.Cropping2D(((0,1),(0,0)), name='Crop_Y')(spec_y)
    #-----------------------------------------------------------------------------

    #ENCODER
    enc = tfkl.Conv2D(16, kernel_size=(3,3), padding='SAME', activation='relu', input_shape=(64,64,1), name='NET1')(spec_x)
    #enc = tfkl.Conv2D(16, kernel_size=(3,3), padding='SAME', activation='relu', name='NET2')(enc)####
    enc = tfkl.Conv2D(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET3')(enc)
    #enc = tfkl.Conv2D(32, kernel_size=(3,3), padding='SAME', activation='relu', name='NET4')(enc)####
    enc = tfkl.Conv2D(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET5')(enc)
    #enc = tfkl.Conv2D(32, kernel_size=(3,3), padding='SAME', activation='relu', name='NET6')(enc)####

    #DECODER
    dec = tfkl.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET7')(enc)
    #dec = tfkl.Conv2D(32, kernel_size=(3,3), padding='SAME', activation='relu', name='NET8')(dec)###
    dec = tfkl.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET9')(dec)
    #dec = tfkl.Conv2D(16, kernel_size=(3,3), padding='SAME', activation='relu', name='NET10')(dec)###
    dec = tfkl.Conv2D(1, kernel_size=(3,3), padding='SAME', activation='relu', name='NET11')(dec)

    estimated = tfkl.Multiply()([dec,spec_x])
    estimated = tf.pad(estimated,((0,0),(0,1),(0,1),(0,0)), name='SALIDA')


    err = MSE()([estimated,spec_y])

    stft_calc = tf.keras.Model(inputs=[x,y],outputs=[err])
    return stft_calc

def mean_loss(y_true, y_pred):
    """custom loss para el ejemplo de denoising
    """
    return tf.reduce_mean(y_pred)

