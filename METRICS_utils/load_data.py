import numpy as np
import os
import cv2
from skimage.transform import resize
from numpy import asarray


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def load_data_from_dirs(real_dir, fake_dir):
    

    real_filenames = [file_name for file_name in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, file_name))]
    fake_filenames = [file_name for file_name in os.listdir(fake_dir) if os.path.isfile(os.path.join(fake_dir, file_name))]

    # loading an image in order to avoid hard-coding the heigh, width and channel
    real_aux_img = cv2.imread(os.path.join(real_dir, real_filenames[0]))
    fake_aux_img = cv2.imread(os.path.join(fake_dir, fake_filenames[0]))

    # (amount of imgs, height, width, channels)
    x_real = np.empty(shape=(len(real_filenames), real_aux_img.shape[0], real_aux_img.shape[1], fake_aux_img.shape[2]))
    x_real = x_real.astype(np.uint8)
    x_fake = np.empty(shape=(len(fake_filenames), fake_aux_img.shape[0], fake_aux_img.shape[1], fake_aux_img.shape[2]))
    x_fake = x_fake.astype(np.uint8)
    
    
    
    # real
    print("Loading real images")
    for img_id,i in zip(real_filenames,range(len(real_filenames))):
      img_uri = os.path.join(real_dir, img_id)
      img = cv2.imread(img_uri)
      x_real[i] = img

    # fake
    print("Loading fake images")
    for img_id,i in zip(fake_filenames,range(len(fake_filenames))):
      img_uri = os.path.join(fake_dir, img_id)
      img = cv2.imread(img_uri)
      x_fake[i] = img

    return x_real, x_fake