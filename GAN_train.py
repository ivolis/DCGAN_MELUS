
import sys

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Numpy
import numpy as np

# Randomness
from numpy.random import random

# Data loading
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 para ver warnings cuda

# Own methods
from GAN_utils import data_management, GAN_models

#for consistency of random numbers and our images
np.random.seed(10)  # numpy library
tf.random.set_seed(10) # tensorflow uses MOSTLY numpy, just in case.

# PARAMETROS DE CORRIDA
output_folder = str(sys.argv[2])
if not os.path.exists(output_folder):
   os.makedirs(output_folder)
per_epoch_visual_control_folder = os.path.join(output_folder, "per_epoch")
os.makedirs(per_epoch_visual_control_folder)
   


# change the below values to the dimensions of your image. The channels number refers to the number of colors
img_rows, img_cols, channels = 128, 128, 1 # LUS -> greyscale, so channels = 1


# Load training data (meter en funcion)
data_folder = str(sys.argv[1])
x_train = data_management.load_images(data_folder,img_rows, img_cols,channels)


#Initialising Hyper Parameters
noise_dim = 100  # input dimension of random vector - the vector that goes into the generator
batch_size = 16   #How many images do we want to include in each batch
steps_per_epoch = int(np.ceil(len(x_train)/batch_size))  #How many steps do we want to take per iteration of our training set (number of batches)
epochs = 1000      #How many iterations of our training set do we want to do.



## MODELO COMPLETO

# Discriminador
discriminator = GAN_models.create_discriminator_cgan(img_cols, img_rows, channels)
discriminator._name = "Discriminador"
discriminator.summary()

# Generador
generator = GAN_models.create_generator_cgan(noise_dim, channels)
generator._name = "Generador"
generator.summary()

# Creo la red GAN
discriminator.trainable = False

gan_input = Input(shape=(noise_dim,))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output)
gan._name = "GAN_completa"
gan.summary()
optimizer_gen = Adam(0.0001, 0.5)
gan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)



# METRICS

# Loss
g_loss_step = np.zeros(len(range(steps_per_epoch)))

g_loss_avg = []
d_loss_real_step = np.zeros(len(range(steps_per_epoch)))
d_loss_fake_step = np.zeros(len(range(steps_per_epoch)))

d_loss_real_avg = []
d_loss_fake_avg = []

# Accuracy
d_acc_real_step = np.zeros(len(range(steps_per_epoch)))
d_acc_fake_step = np.zeros(len(range(steps_per_epoch)))

d_acc_real_avg = []
d_acc_fake_avg = []


for epoch in range(epochs):
    
    # loop temporario para guardar 5imgs/epoch
    aux_noise = np.random.normal(0, 1, size=(5, noise_dim))
    aux_generated_images = generator.predict(aux_noise, verbose = 0)   #Create the images from the GAN.    
    for k, aux_image in enumerate(aux_generated_images):
        save_file_route_name = os.path.join(per_epoch_visual_control_folder , 'generated_image_{:04d}_{:02d}.png'.format(epoch,k))
        plt.imsave(save_file_route_name, aux_image.reshape((img_rows, img_cols))* 127.5 + 
        127.5, cmap = 'gray')
    
    for batch in range(steps_per_epoch):

        # Entreno discriminador
        noise = np.random.normal(0, 1, size=(batch_size, noise_dim))
        fake_x = generator.predict(noise, verbose = 0)

        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        # One Sided Label smoothing: real = [0.9 - 1.1] and fake = 0
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_x, np.zeros(batch_size))
        d_loss_real, d_acc_real = discriminator.train_on_batch(real_x, np.ones(batch_size))

        d_loss = (d_loss_real + d_loss_fake) / 2. # lo dejo solo para el print del final
        #d_loss_step[batch] = d_loss
        d_loss_real_step[batch] = d_loss_real
        d_loss_fake_step[batch] = d_loss_fake

        d_acc_real_step[batch] = d_acc_real
        d_acc_fake_step[batch] = d_acc_fake

        # Entreno generador
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch(noise, y_gen)
        g_loss_step[batch] = g_loss

    print(f'Epoch: {epoch + 1} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

    d_loss_real_avg.append(np.mean(d_loss_real_step))
    d_loss_fake_avg.append(np.mean(d_loss_fake_step))
    g_loss_avg.append(np.mean(g_loss_step))
    d_acc_real_avg.append(np.mean(d_acc_real_step))
    d_acc_fake_avg.append(np.mean(d_acc_fake_step))




# Guardo los resultados de la corrida

# Guardo los resultados de la corrida
import pandas as pd

data = {"epoch" : range(1,epochs+1),
	"discriminator_loss_real": d_loss_real_avg,
	"discriminator_loss_fake": d_loss_fake_avg,
	"generator_loss": g_loss_avg,
	"discriminator_acc_real": d_acc_real_avg,
	"discriminator_acc_fake": d_acc_fake_avg,
	}	
df = pd.DataFrame(data)

df.to_csv("results.csv", index = False)


# Guardo las imagenes
save_imgs_amount = int(sys.argv[3])
data_management.save_images(save_imgs_amount, output_folder, noise_dim, generator, img_rows, img_cols, channels)