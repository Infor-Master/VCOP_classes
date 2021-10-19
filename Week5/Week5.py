import matplotlib.pyplot as plt
import numpy as np
import cv2

def ex_1_1_log(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(1 + image))
    log_image = np.array(log_image, dtype=np.uint8)
    plt.imshow(log_image)
    plt.show()


def ex_1_2_gamma(image, gamma_value):
    table = np.arange(0, 256)/255
    table = np.power(table, gamma_value)
    table = np.uint8(table*255)
    plt.plot(table)
    plt.show()

    gamma_image = cv2.LUT(image, table)
    plt.imshow(gamma_image)
    plt.show()

def ex_1_3_negative(image):
    #neg_image = 255 - image
    neg_image = cv2.bitwise_not(image)

    plt.imshow(neg_image)
    plt.show()

def ex_1_4_gray_level(image, Tmin, Tmax, Vtrue= None, Vfalse = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    m,n = image.shape
    L = image.max()
    neg_image = L - image
    tresh_image = np.zeros((m,n), dtype = int)

    for i in range(m):
        for j in range(n):
            if Tmin < image[i,j] < Tmax:
                if Vtrue == None:
                    tresh_image[i,j] = image[i,j]
                else:
                    tresh_image[i, j] = Vtrue
            else:
                if Vfalse == None:
                    tresh_image[i, j] = image[i, j]
                else:
                    tresh_image[i, j] = Vfalse
    plt.imshow(tresh_image, cmap="gray")
    plt.show()

def ex_2_1_gray_scale_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # --------- opencv
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram)
    # --------- matplotlib
    # plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def ex_2_2_gray_scale_histogram_equalize(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalize_image = cv2.equalizeHist(image)
    plt.imshow(equalize_image, cmap="gray")
    plt.show()
    res = np.hstack((image, equalize_image))
    plt.imshow(res, cmap="gray")
    plt.show()

image = cv2.imread("./img.png")
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
plt.imshow(image_bgr)
plt.show()


ex_1_1_log(image_bgr)
ex_1_2_gamma(image_bgr, 5)
ex_1_3_negative(image_bgr)
ex_1_4_gray_level(image_bgr, Tmin=0, Tmax=150, Vtrue=0, Vfalse=255)
ex_1_4_gray_level(image_bgr, Tmin=100, Tmax=180, Vtrue=255)
ex_2_1_gray_scale_histogram(image_bgr)
ex_2_2_gray_scale_histogram_equalize(image_bgr)

