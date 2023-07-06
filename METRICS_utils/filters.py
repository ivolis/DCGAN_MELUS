import numpy as np
from skimage.filters import gaussian
from skimage import img_as_ubyte
from scipy.stats import truncnorm



def apply_gaussian_blur(imgs,sigma):
    transf_data = np.zeros_like(imgs)
    for i in range(0,len(imgs)):
        transf_data[i] = img_as_ubyte(gaussian(imgs[i], sigma))
    return transf_data


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def apply_gauss_noise(imgs, alpha):
    transf_data = np.zeros_like(imgs)
    for i in range(0,len(imgs)):
        rnd_aux = get_truncated_normal(255/2, 255/6,0,255)
        rnd = rnd_aux.rvs(imgs.shape[1] * imgs.shape[2] * imgs.shape[3])
        rnd = rnd.reshape(imgs.shape[1] , imgs.shape[2] , imgs.shape[3])
        if alpha > 1e-6:
            transf_data[i] = (1-alpha)*imgs[i] + alpha*rnd
        else:
            transf_data[i] = imgs[i].copy()
    return transf_data


def apply_salt_and_pepper(imgs, p):
    transf_data = imgs.copy()
    h = imgs.shape[1]
    w = imgs.shape[2]
    c = imgs.shape[3]
    salt = -1
    pepper = 1
    for k in range(len(imgs)):
        ns, d0, d1, d2 = transf_data[k].reshape(-1,h,w,c).shape
        coords = np.random.rand(ns,d0,d1) < p
        n_co = coords.sum()
        vals = (np.random.rand(n_co) < 0.5).astype(np.float32)
        vals[vals < 0.5] = salt; vals[vals > 0.5] = pepper
        for i in range(c):
            transf_data[k].reshape(-1,h,w,c)[coords,i] = vals
    return transf_data
