from keras.initializers import RandomNormal
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# GENERATOR #
def create_generator_cgan(noise_dim, channels):
    
    generator = Sequential()

    d = 16
    generator.add(Dense(d*d*256, kernel_initializer=RandomNormal(0, 0.02), input_dim=noise_dim))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Reshape((d, d, 256)))

    generator.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU(0.2))



    generator.add(Conv2D(channels, (5, 5), padding='same', activation='tanh', kernel_initializer=RandomNormal(0, 0.02))) 

    return generator




# DISCRIMINATOR #
def create_discriminator_cgan(img_cols, img_rows, channels):
    
    optimizer_disc = Adam(0.00001, 0.5)

    discriminator = Sequential()
    
    discriminator.add(Conv2D(64, (5, 5), padding='same', kernel_initializer=RandomNormal(0, 0.02), input_shape=(img_cols, img_rows, channels)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(128, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Conv2D(256, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Flatten())
    discriminator.add(Dropout(0.4))
    discriminator.add(Dense(1, activation='sigmoid', input_shape=(img_cols, img_rows, channels)))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_disc, metrics=['accuracy'])
    return discriminator