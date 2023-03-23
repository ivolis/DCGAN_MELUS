import numpy as np
import os
from IPython.display import clear_output
import cv2



def load_data_MELUS(real_dir, fake_dir):
  
    # LUS_img_type = "nCOVID"
    # real_img_filename_start = "nNL"
    # fake_img_filename_start = "generated_image"

    

    real_filenames = [file_name for file_name in os.listdir(real_dir) if file_name.startswith("nNL")]
    fake_filenames = [file_name for file_name in os.listdir(fake_dir) if file_name.startswith("generated_image")]
    # agarro dos imagenes random para no hc height y width
    real_aux_img = cv2.imread(real_dir + real_filenames[0])
    fake_aux_img = cv2.imread(fake_dir + fake_filenames[0])

    # (amount of imgs, height, width, channels)
    x_real = np.empty(shape=(len(real_filenames), real_aux_img.shape[0], real_aux_img.shape[1], fake_aux_img.shape[2]))
    x_real = x_real.astype(np.uint8)
    x_fake = np.empty(shape=(len(fake_filenames), fake_aux_img.shape[0], fake_aux_img.shape[1], fake_aux_img.shape[2]))
    x_fake = x_fake.astype(np.uint8)
    
    
    # se puede mejorar en vez de meter 2 loops?
    
    # real
    for img_id,i in zip(real_filenames,range(len(real_filenames))):
      img_uri = real_dir + img_id
      #print("Subiendo imagen real %5d de %5d " % (i,len(real_filenames)))
      img = cv2.imread(img_uri)
      x_real[i] = img
      clear_output(True)

    # fake
    for img_id,i in zip(fake_filenames,range(len(fake_filenames))):
      img_uri = fake_dir + img_id
      #print("Subiendo imagen falsa %5d de %5d " % (i,len(fake_filenames)))
      img = cv2.imread(img_uri)
      x_fake[i] = img
      clear_output(True)

    return x_real, x_fake