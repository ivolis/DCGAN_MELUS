import tensorflow as tf
from tensorflow import keras
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import sys


generator = keras.models.load_model(str(sys.argv[1]))
amount_images = int(sys.argv[2])
# generator.summary()

# La primera capa del generador es la entrada, el input shape es (None, dim espacio latente)
noise_dim = generator.layers[0].input_shape[1]

noise = np.random.normal(0, 1, size = (amount_images, noise_dim))
generated_images = generator.predict(noise, verbose = 0)

for k, image in enumerate(generated_images):
    plt.imsave('image_{:04d}.png'.format(k), image.reshape(128,128)*127.5 + 127.5, cmap = 'gray')