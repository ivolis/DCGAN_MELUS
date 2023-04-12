import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def load_images(folder_dir, img_rows, img_cols, channels):

    relevant_files = [file_name for file_name in os.listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, file_name))]
    total_images = len(relevant_files)

    x_train = np.empty(shape=(total_images, img_rows, img_cols))

    for img_id,i in zip(relevant_files,range(total_images)):
        img_uri = os.path.join(folder_dir, img_id)
        img = cv2.imread(img_uri, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        x_train[i] = img

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5       #Normalize the images again so that the pixel value is from -1 to 1

    x_train = x_train.reshape(-1, img_rows, img_cols, channels)  #Reshaping the data into a more NN friendly format
    
    return x_train



def save_images(amount, folder_name, noise_dim, generator, img_rows, img_cols, channels):
    for k in range(amount):
        noise = np.random.normal(0, 1, size=(1, noise_dim))
        generated_images = generator.predict(noise, verbose = 0)   #Create the images from the GAN.    
        for i, image in enumerate(generated_images):
            save_file_route_name = os.path.join(folder_name, 'generated_image_{:04d}.png'.format(k))
            if channels == 1:
                plt.imsave(save_file_route_name, image.reshape((img_rows, img_cols))* 127.5 + 
            127.5, cmap='gray')
            else:
                plt.imsave(save_file_route_name, image.reshape((img_rows, img_cols))* 127.5 + 
            127.5)
