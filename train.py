"""
Bucle de entrenamiento
"""
import numpy as np
import tensorflow as tf
from model.autoencoder import basic_autoencoder

modelo = basic_autoencoder()
modelo.summary()
#history = modelo.fit(training_generator, epochs= 10)
#modelo.fit([train_noisy,train_clean],[train_noisy], batch_size=64, epochs=20)
