import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

from sklearn.metrics.pairwise import polynomial_kernel


def calculate_metrics(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    fid = calculate_fid(act1,act2)
    kid = calculate_kid(act1,act2)

    return fid, kid

def calculate_fid(act1,act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_kid(act1,act2):

    # KID (MMD2 usando activaciones Inception con kernel polinomico) basado en su definicion 
    m = act1.shape[0]
    n = act2.shape[0]

    gram_1 = polynomial_kernel(act1, degree=3,coef0=1) # kernel
    gram_1 = gram_1 - np.diag(np.diagonal(gram_1)) # en la sumatoria i=/=j (resto su diagonal)

    gram_2 = polynomial_kernel(act1, degree=3,coef0=1)
    gram_2 = gram_2 - np.diag(np.diagonal(gram_2))

    sum_1 = np.sum(gram_1)
    sum_2 = np.sum(gram_2)

    kid = sum_1 / (m*(m-1)) + sum_2 / (n*(n-1)) - np.sum(polynomial_kernel(act1,act2,degree=3,coef0=1)) * (2/(m*n))

    return kid


