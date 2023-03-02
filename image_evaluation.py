import numpy as np
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
import sys

# Own methods
from load_data import load_data_MELUS
from FID import scale_images, calculate_fid


# python3 image_evaluation.py FAKE_IMAGES_DIRECTORY
real_dir = "DB LUS4MELUS/tif/"
fake_dir = str(sys.argv[1])

images1, images2 = load_data_MELUS(real_dir, fake_dir)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# Shuffle the images
shuffle(images1)
shuffle(images2)
print('Loaded', images1.shape, images2.shape)

# convert integer to floating point values
images1 = images1.astype('float32')
images2 = images2.astype('float32')

# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)

# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)