import numpy as np
from scipy.fft import fft2, ifftshift, ifft2, fftshift
import math


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    radius = np.arange(size) - (size - 1) / 2
    gaus_kernel = np.exp(-(radius ** 2) / (2 * sigma ** 2))
    gaus_kernel = np.outer(gaus_kernel, gaus_kernel)

    return gaus_kernel / gaus_kernel.sum()


# from seminar notebook
def pad_kernel(kernel, target):
    th, tw = target
    kh, kw = kernel.shape[:2]
    ph, pw = th - kh, tw - kw

    padding = [((ph + 1) // 2, ph // 2), ((pw + 1) // 2, pw // 2)]
    kernel = np.pad(kernel, padding)

    return kernel
    

def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    padded_h = pad_kernel(h, shape)
    return fft2(ifftshift(padded_h))


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.copy(H)
    H_inv[np.abs(H) <= threshold] = 0
    H_inv[np.abs(H) > threshold] = 1. / H[np.abs(H) > threshold]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    F = G * H_inv
    f = fftshift(ifft2(F))
    return f.real


def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img, blurred_img.shape)
    H = fourier_transform(h, blurred_img.shape)
    H_ = np.conj(H)
    H_abs = H_ * H
    F = H_ / (H_abs + K) * G
    f = fftshift(ifft2(F))
    return f.real


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    MAX = 255
    
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = ((img1 - img2) ** 2).mean()
    
    return 20 * math.log10(MAX / np.sqrt(mse))
