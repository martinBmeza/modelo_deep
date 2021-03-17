"""Modelo de red utilizado"""

import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np 

eps = np.finfo(float).eps

class Spectrogram(tfkl.Layer):
    def __init__(self,win_size,hop_size,fft_size=None,calculate='magnitude',window=tf.signal.hann_window,pad_end=False,name=None, trainable=False):
        super(Spectrogram, self).__init__(name=name)

        self.stft_args = {'ws': win_size,
                  'hs': hop_size,
                  'ffts': fft_size,
                  'win': window,
                  'pad': pad_end,
                  'calculate': calculate}

    def call(self,x):
        stft = tf.signal.stft(
                signals=x,
                frame_length=self.stft_args['ws'],
                frame_step=self.stft_args['hs'],
                fft_length=self.stft_args['ffts'],
                window_fn=self.stft_args['win'],
                pad_end=self.stft_args['pad'])

        calculate = self.stft_args['calculate']
        if calculate == 'magnitude':
            return tf.abs(stft)
        elif calculate == 'complex':
            return stft
        elif calculate == 'phase':
            return tf.math.angle(stft)
        else:
            raise Exception("{} not recognized as calculate parameter".format(calculate))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'win_size': self.stft_args['ws'],
            'hop_size': self.stft_args['hs'],
            'fft_size': self.stft_args['ffts'],
            'calculate': self.stft_args['calculate'],
            'window': self.stft_args['win'],
            'pad_end': self.stft_args['pad']
        })
        return config

class SoftMask(tfkl.Layer):
    def __init__(self,name=None):
        super(SoftMask,self).__init__(name=name)

    def call(self,x):
        sources = x[0]
        to_mask = x[1]
        total_sources = tf.expand_dims(tf.reduce_sum(sources,axis=1),axis=1)
        mask = sources/(total_sources+1e-9)
        to_mask = tf.expand_dims(to_mask,axis=1)
        return mask*to_mask

class TranslateRange(tfkl.Layer):
    def __init__(self,name=None,trainable=False,original_range=None,target_range=None):
        super(TranslateRange,self).__init__(name=name)
        self.original_range = original_range
        self.target_range = target_range

    def call(self,x):
        offset = self.target_range[0] - self.original_range[0]
        x_range = self.original_range[1] - self.original_range[0]
        y_range = self.target_range[1] - self.target_range[0]

        return y_range*(x + offset)/x_range
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'original_range': self.original_range,
            'target_range': self.target_range,
        })
        return config

class MSE(tfkl.Layer):
    #x[0]: predicted
    #x[1]: original
    def __init__(self,name=None,trainable=False,lnorm=2,offset=1e-9,normalize=False):
        super(MSE,self).__init__(name=name)
        self.offset = offset
        self.normalize = normalize
        self.lnorm = lnorm
    
    def call(self,x):
        mse_error = tf.abs(x[0] - x[1])**self.lnorm
        if self.normalize:
            mse_error = mse_error/(self.offset + tf.abs(x[1])**self.lnorm)
        return mse_error
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'offset': self.offset,
            'normalize': self.normalize,
            'lnorm' : self.lnorm
        })
        return config
    
def mean_loss(y_true, y_pred):
  return tf.reduce_mean(y_pred)

def mean_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def basic_autoencoder():
    
    x = tfkl.Input((32000), name = 'Entrada_Noisy')
    y = tfkl.Input((32000), name = 'Target_Clean')
       
    #GENERACION DE ESPECTROS
    spec_x = Spectrogram(1024,512,name='STFT_X')(x)
    spec_x = tf.math.log(spec_x+eps)
    spec_x = TranslateRange(original_range=[-5, 5],target_range=[0,1.0])(spec_x)
    spec_x = tf.expand_dims(spec_x,axis=-1, name= 'expand_X')
    spec_x = tfkl.Cropping2D(((0,1),(0,1)), name = 'Crop_X')(spec_x)
    
    spec_y = Spectrogram(1024,512,name='STFT_Y')(y)
    spec_y = tf.expand_dims(spec_y,axis=-1, name='expand_Y')
    
    #ENCODER
    enc = tfkl.Conv2D(16, kernel_size=(3,3), padding='SAME', activation='relu', input_shape=(64,64,1), name='NET1')(spec_x)
    enc = tfkl.Conv2D(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET3')(enc)
    enc = tfkl.Conv2D(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET5')(enc)
    
    #DECODER
    dec = tfkl.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET7')(enc)
    dec = tfkl.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='NET9')(dec)
    dec = tfkl.Conv2D(1, kernel_size=(3,3), padding='SAME', activation='relu', name='NET11')(dec)
    
    estimated = tfkl.Multiply()([dec,spec_x])
    estimated = tf.pad(estimated,((0,0),(0,1),(0,1),(0,0)), name='SALIDA')
    err = MSE()([estimated,spec_y])
    stft_calc = tf.keras.Model(inputs=[x,y],outputs=[err])
    
    
    stft_calc.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=mean_loss)
    
    return stft_calc

