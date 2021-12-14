import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

def prep_array(M, K, A):
    points = np.zeros(M)
    points[:K] = A
    return points

def prep_delta_array(M, T, A):
    points = np.zeros(M)
    for indx in range(len(points)):
        if indx%T == 0:
            points[indx] = A
    return points

def prep_sin_array(M, F):
    in_array = np.linspace(np.pi/2, (np.pi*F*2)+(np.pi/2), M)
    out_array = np.sin(in_array)
    return out_array

def fourier_1D_spectrum(arr):
    f = np.fft.fft(arr)
    fshift = np.fft.fftshift(f)
    spectrum = np.abs(fshift)

    fig, axs = plt.subplots(2)
    axs[0].plot(arr)
    axs[1].plot(spectrum)
    plt.show()
    return spectrum

def fourier_2D_spectrum(img):

    ft = np.fft.fft2(img)
    fshift = np.fft.fftshift(ft)
    spectrum = np.abs(fshift)

    plt.set_cmap("gray")
    fig, axs = plt.subplots(2)
    axs[0].imshow(img)
    axs[1].imshow(spectrum)
    plt.show()
    plt.close()
    return spectrum

def gen_rect_image(size, x, y):
    image = np.zeros((size, size))
    dif_X = (size - x)/2
    dif_Y = (size - y)/2
    for i in range(size):
        if (i > dif_X and i <= dif_X + x):
            for j in range(size):
                if (j > dif_Y and j <= dif_Y + y):
                    image[i, j] = 1
    return image

def gen_wave_image(size, wavelength, angle):
    x = np.arange(-size/2, size/2, 1)
    X, Y = np.meshgrid(x, x)
    image = np.sin(
        2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
    )
    return image

def gen_impulse_image(size, distance):
    image = np.zeros((size, size))
    for i in range(size):
        if (i%distance==0):
            for j in range(size):
                if (j%distance==0):
                    image[i, j] = 1
    return image

def ideal_lowpass_filtering(img, r):
    ft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(ft)
    mag = np.abs(dft_shift)

    mask = np.zeros_like(img)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    circ = cv2.circle(mask, (cx, cy), r, (255, 255, 255), -1)[0]
    # a funcao circle afecta diretamente na imagem passada (mask). O retorno não é o que se deseja
    dft_shift_masked = np.multiply(dft_shift, mask) / 255
    back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
    img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0, 1))
    img_filtered = np.abs(img_filtered).clip(0, 255).astype(np.uint8)

    plt.set_cmap("gray")
    fig, axs = plt.subplots(2)
    axs[0].imshow(img)
    axs[1].imshow(img_filtered)
    plt.show()
    plt.close()

def gaussian_highpass_filtering(img, d):
    lowpass = ndimage.gaussian_filter(img, d)
    highpass = np.subtract(img, lowpass)
    plt.set_cmap("gray")
    fig, axs = plt.subplots(2)
    axs[0].imshow(img)
    axs[1].imshow(highpass)
    plt.show()
    plt.close()

fourier_1D_spectrum(prep_array(1000, 10, 1))
fourier_1D_spectrum(prep_array(1000, 20, 1))
fourier_1D_spectrum(prep_array(1000, 40, 1))
fourier_1D_spectrum(prep_array(1000, 1000, 1))
fourier_1D_spectrum(prep_array(1000, 1, 1))
fourier_1D_spectrum(prep_delta_array(1000, 50, 1))
fourier_1D_spectrum(prep_sin_array(100, 3))
fourier_1D_spectrum(prep_sin_array(100, 10))

fourier_2D_spectrum(gen_rect_image(100, 5, 20))
fourier_2D_spectrum(gen_wave_image(100, 10, 0))
fourier_2D_spectrum(gen_wave_image(100, 10, np.pi/4))
fourier_2D_spectrum(gen_rect_image(100, 8, 8))
fourier_2D_spectrum(cv2.GaussianBlur(gen_rect_image(100, 8, 8), (25, 25), 0))
fourier_2D_spectrum(gen_impulse_image(100, 20))

img_0333 = cv2.imread("./Fig0333_a__test_pattern_blurring_orig_.tif")
img_0333 = cv2.cvtColor(img_0333, cv2.COLOR_RGB2GRAY)

ideal_lowpass_filtering(img_0333, 5)
ideal_lowpass_filtering(img_0333, 15)
ideal_lowpass_filtering(img_0333, 30)
ideal_lowpass_filtering(img_0333, 80)
ideal_lowpass_filtering(img_0333, 230)
gaussian_highpass_filtering(img_0333, 15)
gaussian_highpass_filtering(img_0333, 30)
gaussian_highpass_filtering(img_0333, 80)



